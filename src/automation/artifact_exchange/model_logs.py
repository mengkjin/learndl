"""Append-only model logs and idempotent receiver commits."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from datetime import datetime, timezone

from .core import atomic_json, component, fsync_dir, real_tree

__all__ = ['STAGES', 'now', 'log_root', 'append_received', 'capture_run']


STAGES = ('train', 'test')


def now():
    return datetime.now(timezone.utc).isoformat()


def log_root(config, host, model):
    return real_tree(config.results / 'external/models' / component(host) / component(model) / 'run_logs')


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


@contextmanager
def capture_run(trainer):
    """Keep optional HTML capture and collection scoped to a single invocation."""
    from types import SimpleNamespace
    from .research import CURRENT_TASK, model_run
    from src.proj import Save

    owner = CURRENT_TASK.get()
    run_id = trainer.training_run.run_id
    log = SimpleNamespace(run_id=run_id, html=None)
    if owner is not None:
        owner._model_run_ids.append(run_id)
    catcher = nullcontext()
    if title := getattr(trainer, 'model_html_title', None):
        from src.proj.util.catcher import HtmlCatcher

        log.html = trainer.base_path('run_logs', 'html', f'{run_id}.html')
        catcher = HtmlCatcher(title=title)
        catcher.set_export_files(log.html)
    with model_run(trainer, log):
        try:
            with catcher:
                yield
                Save.async_wait_all(caller_name='model_run_log')
        finally:
            if log.html is not None and log.html.is_file() and owner is not None:
                from src.proj import Proj
                Proj.exit_files.append(log.html)
