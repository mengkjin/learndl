"""Durable records, bounded output capture and notifications for CLI reconstructions."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from .diagnostic_sampler import alive, atomic_json, evidence, identity

SOURCE = 'cli-reconstruct'


def runtime_dir() -> Path:
    from src.proj import PATH
    return PATH.runtime / 'interactive_runs'


class Tee:
    """Stream to disk first, without retaining output or depending on a live terminal."""
    def __init__(self, original, output):
        self.original, self.output = original, output

    def write(self, text):
        self.output.write(text)
        self.output.flush()
        try:
            self.original.write(text)
            self.original.flush()
        except (OSError, ValueError):
            pass
        return len(text)

    def flush(self):
        self.output.flush()

    def isatty(self):
        return self.original.isatty()

    def __getattr__(self, name):
        return getattr(self.original, name)


def tail(path: Path, size: int = 8192) -> str:
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read(size).decode('utf-8', errors='replace')
    except OSError as exc:
        return f'Log unavailable: {exc}'


def start_sampler(directory: Path, process: dict) -> dict:
    script = Path(__file__).with_name('diagnostic_sampler.py')
    argv = [sys.executable, str(script), str(directory), str(process['pid']), str(process['created'])]
    if sys.platform == 'linux':
        env = dict(os.environ)
        env.setdefault('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
        env.setdefault('DBUS_SESSION_BUS_ADDRESS', f'unix:path={env["XDG_RUNTIME_DIR"]}/bus')
        try:
            result = subprocess.run(['systemd-run', '--user', '--quiet', '--collect',
                                     f'--unit=learndl-sample-{directory.name}', '--', *argv],
                                    env=env, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return {'mode': 'user-service'}
            failure = result.stderr[-2048:]
        except (OSError, subprocess.SubprocessError) as exc:
            failure = str(exc)
    else:
        failure = 'systemd user services unavailable on this platform'
    # Useful outside Linux too; explicitly expose the weaker cgroup isolation.
    with (directory / 'sampler-stderr.log').open('a') as errors:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=errors, start_new_session=True)
    return {'mode': 'detached-process', 'pid': child.pid, 'isolation_warning': failure}


@contextmanager
def reconstruction(processor):
    """Register before computation, keep complete output on disk even after SIGKILL."""
    from src.api.util.backend.recorder import BackendTaskRecorder
    from src.api.util.backend.task import TaskDatabase, TaskItem
    from src.proj import PATH
    from src.proj.util.cli.session import git_head

    run_id = uuid.uuid4().hex
    directory = runtime_dir() / run_id
    directory.mkdir(parents=True, mode=0o700)
    # TaskItem uses script-relative names as its identity. These are virtual task
    # labels, not runnable pipeline scripts; UUID avoids its seconds-only ID collision.
    label = PATH.scpt / '_interactive' / f'reconstruct-{processor.key}-{processor.frame}-{run_id}.py'
    db = TaskDatabase()
    task = TaskItem.create(label, db, source=SOURCE)
    log = directory / 'output.log'
    log.touch()
    metadata = {'task_id': task.id, 'key': processor.key, 'frame': processor.frame,
                'start': int(processor.load_start), 'end': int(processor.load_end),
                'revision': git_head(PATH.main), 'process': identity(os.getpid()),
                'started': time.time(), 'log': str(log), 'host': __import__('socket').gethostname()}
    atomic_json(directory / 'run.json', metadata)
    files = [str(log), str(directory / 'run.json'), str(directory / 'memory.jsonl')]
    db.update_task(task.id, cmd=f'reconstruct {processor.key} --frame {processor.frame}', exit_files=files)
    recorder = BackendTaskRecorder(task_id=task.id, parse_cli=False)
    recorder.task_db = db
    previous_trace = os.environ.get('LEARNDL_MEMORY_TRACE')
    with recorder:
        with log.open('a', buffering=1) as output, redirect_stdout(Tee(sys.stdout, output)), redirect_stderr(Tee(sys.stderr, output)):
            print(json.dumps(metadata, ensure_ascii=False))
            try:
                os.environ['LEARNDL_MEMORY_TRACE'] = '1'
                try:
                    metadata['sampling'] = start_sampler(directory, metadata['process'])
                except (OSError, subprocess.SubprocessError) as exc:
                    metadata['sampling'] = {'unavailable': str(exc)}
                atomic_json(directory / 'run.json', metadata)
                print('Memory sampler:', metadata['sampling'])
                yield task.id
            except BaseException:
                traceback.print_exc()
                atomic_json(directory / 'finished.json', {'status': 'error', 'time': time.time()})
                raise
            else:
                atomic_json(directory / 'finished.json', {'status': 'complete', 'time': time.time()})
            finally:
                if previous_trace is None:
                    os.environ.pop('LEARNDL_MEMORY_TRACE', None)
                else:
                    os.environ['LEARNDL_MEMORY_TRACE'] = previous_trace


def maintain(task_db, email_sender, *, directory: Path | None = None, now: float | None = None) -> bool:
    """Durable failure outbox, external evidence recovery and 30-day retention.

    Database reconciliation runs before this function in the watchdog. Only our
    source is handled here; generic script exception emails remain unchanged.
    """
    directory = directory or runtime_dir()
    now = time.time() if now is None else now
    delivered = True
    if not directory.exists():
        return True
    for path in directory.glob('*/run.json'):
        folder = path.parent
        try:
            meta = json.loads(path.read_text())
            task = task_db.get_task(meta['task_id'])
            if task is None or task.status not in {'error', 'killed', 'complete'}:
                continue
            ended = task.end_time or now
            if task.status in {'error', 'killed'} and not (folder / 'notified.json').exists():
                if task.status == 'killed' and not (folder / 'evidence.json').exists():
                    # Recover even when the sampler was killed along with the terminal.
                    group = None
                    try:
                        sampler = json.loads((folder / 'sampler.json').read_text())
                        if sampler.get('cgroup'):
                            group = Path(sampler['cgroup'])
                    except (OSError, ValueError):
                        pass
                    atomic_json(folder / 'evidence.json', evidence(meta['started'], meta['process'], group))
                files = [str(p) for p in (folder / 'output.log', path, folder / 'memory.jsonl', folder / 'evidence.json') if p.exists()]
                task_db.update_task(task.id, exit_files=files)
                body = (f'Task: {task.id}\nHost: {meta.get("host")}\n'
                        f'Reconstruct: {meta["key"]} / {meta["frame"]}\nStatus: {task.status}\n'
                        f'Range: {meta["start"]}-{meta["end"]}\nRevision: {meta.get("revision")}\n'
                        f'Logs: {folder}\nError: {task.exit_error}\n'
                        'Unexpected termination alone does not confirm OOM.\n\n' + tail(folder / 'output.log'))
                atomic_json(folder / 'pending.json', {'task_id': task.id, 'status': task.status, 'body': body})
                try:
                    sent = email_sender(f'Learndl reconstruction {task.status}: {meta["key"]}/{meta["frame"]}',
                                        body, attachments=[], confirmation_message='Interactive reconstruction alert')
                except Exception:
                    sent = False
                if sent:
                    atomic_json(folder / 'notified.json', {'time': now, 'status': task.status})
                    (folder / 'pending.json').unlink(missing_ok=True)
                else:
                    delivered = False
            # Never remove logs for a live process, an unsent failure or a recent run.
            if now - ended > 30 * 86400 and not alive(meta['process']) and (
                task.status == 'complete' or (folder / 'notified.json').exists()
            ):
                shutil.rmtree(folder)
        except (OSError, ValueError, KeyError):
            delivered = False
    return delivered
