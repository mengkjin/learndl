"""Append-only model logs and idempotent receiver commits."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .core import atomic_json, component, fsync_dir, real_tree

STAGES = ('train', 'test')


def now():
    return datetime.now(timezone.utc).isoformat()


def log_root(config, host, model):
    return real_tree(config.results / 'model_run_logs' / component(host) / component(model))


def append_received(config, target, manifest):
    """Recover an interrupted append using its durable offset before writing again."""
    from contextlib import closing
    import sqlite3

    if manifest.get('category') != 'model_training':
        return
    folder = log_root(config, manifest['host'], manifest['experiment'])
    for phase in STAGES:
        name = f'run-log/{phase}.md'
        if not any(item['path'] == name for item in manifest['files']):
            continue
        fragment = real_tree(target / name)
        content = fragment.read_bytes()
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / 'model.json', {'host': manifest['host'], 'model': manifest['experiment']})
        path = real_tree(folder / f'{phase}.md')
        pending = folder / f'.{phase}.pending.json'
        with closing(sqlite3.connect(folder / '.runs.sqlite3')) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS runs (phase TEXT, id TEXT, sha TEXT, PRIMARY KEY (phase, id))')
        key = (phase, manifest['run_id'])
        sha = hashlib.sha256(content).hexdigest()
        with closing(sqlite3.connect(folder / '.runs.sqlite3')) as db:
            previous = db.execute('SELECT sha FROM runs WHERE phase=? AND id=?', key).fetchone()
            if previous:
                if previous[0] != sha:
                    raise ValueError('Model run ID reused with different log content')
                if pending.exists() and json.loads(pending.read_text())['id'] == key[1]:
                    pending.unlink()
                    fsync_dir(folder)
                continue
            if pending.exists():
                journal = json.loads(pending.read_text())
                if (journal['id'], journal['sha']) != (key[1], sha):
                    raise ValueError('Prior model log append must finish first')
            else:
                journal = {'id': key[1], 'sha': sha, 'offset': path.stat().st_size if path.exists() else 0}
                atomic_json(pending, journal)
            offset = journal['offset']
            with path.open('a+b') as stream:
                if stream.seek(0, os.SEEK_END) < offset:
                    raise ValueError('Model log changed during an unfinished append')
                stream.truncate(offset)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            fsync_dir(folder)
            with db:
                db.execute('INSERT INTO runs VALUES (?, ?, ?)', (*key, sha))
            pending.unlink()
            fsync_dir(folder)


class RunLog:
    def __init__(self, folder: Path, fragments: Path, run_id: str, task_id, operation: str):
        self.folder = folder
        self.fragments = fragments
        self.run_id = run_id
        self.task_id = task_id
        self.operation = operation
        self.started = now()
        self.phases: set[str] = set()
        self.html: Path | None = None

    def write(self, phase: str, message: str):
        header = ''
        if phase not in self.phases:
            header = f'\n## {self.started} · {self.run_id}\n\n任务 `{self.task_id or "独立运行"}` · {self.operation}\n\n'
            self.phases.add(phase)
        self.folder.mkdir(parents=True, exist_ok=True)
        line = header + '- ' + message.replace('\n', ' ') + '\n'
        for folder in (self.folder, self.fragments):
            with (folder / f'{phase}.md').open('a', encoding='utf-8') as stream:
                stream.write(line)
                stream.flush()
                if folder == self.folder:
                    os.fsync(stream.fileno())


@contextmanager
def capture_run(trainer):
    from .research import CURRENT_TASK, model_run

    owner = CURRENT_TASK.get()
    run_id = uuid4().hex
    base = trainer.config.base_path
    operation = getattr(trainer, 'run_operation', 'internal')
    with TemporaryDirectory(prefix='learndl-model-log-') as work:
        log = RunLog(base('run_logs'), Path(work).resolve(), run_id, getattr(owner, 'task_run_id', None), operation)
        if owner is not None:
            owner._model_run_ids.append(run_id)
        trainer.process_log = log
        catcher = nullcontext()
        if title := getattr(trainer, 'model_html_title', None):
            from src.proj.util.catcher import HtmlCatcher

            log.html = base('run_logs', 'html', f'{run_id}.html')
            html_catcher = HtmlCatcher(title=title)
            html_catcher.set_export_files(log.html)
            catcher = html_catcher
        state = 'completed'
        try:
            with model_run(trainer, log):
                try:
                    for stage, phase in (('fit', 'train'), ('test', 'test')):
                        if stage in trainer.queue_of_stages:
                            trainer.write_process_log(phase, f'{now()} · planned')
                    with catcher:
                        yield
                        from src.proj import Save
                        Save.async_wait_all(caller_name='model_run_log')
                except BaseException as exc:
                    state = f'failed ({type(exc).__name__}): {exc}'
                    raise
                finally:
                    if log.html is not None and log.html.is_file() and owner is not None:
                        from src.proj import Proj
                        Proj.exit_files.append(log.html)
                    for stage in ('fit', 'test'):
                        if stage in trainer.queue_of_stages:
                            trainer.write_process_log('train' if stage == 'fit' else 'test', f'{now()} · run {state}')
        finally:
            trainer.process_log = None


def record_skipped(base, operation, phase, reason):
    """Log an operation rejected by the recent-run guard without invoking the trainer."""
    from portalocker.exceptions import LockException

    from .core import ArtifactSpec
    from .research import CURRENT_TASK, publish_safe
    from src.res.model.util.lifecycle import ModelLifecycle
    from src.proj import Logger

    owner = CURRENT_TASK.get()
    run_id = uuid4().hex
    try:
        with ModelLifecycle(base.base).lock(), TemporaryDirectory(prefix='learndl-model-log-') as work:
            log = RunLog(base('run_logs'), Path(work).resolve(), run_id,
                         getattr(owner, 'task_run_id', None), operation)
            for name in phase:
                log.write(name, reason)
                log.write(name, f'{now()} · run skipped')
            if owner is not None:
                owner._model_run_ids.append(run_id)
            publish_safe('model_training', run_id, [
                ArtifactSpec(log.fragments, category='model_training',
                             experiment=base.full_name, destination='run-log')
            ], status='skipped')
    except (OSError, RuntimeError, LockException) as exc:
        Logger.warning(f'Could not record skipped model run: {exc}', vb_level=0)
