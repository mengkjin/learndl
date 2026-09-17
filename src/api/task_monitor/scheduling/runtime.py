"""Durable run identities and two-phase termination of installer-owned runs."""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import time
from pathlib import Path
from typing import Any

import psutil

from .config import runtime_dir


class RunStore:
    def __init__(self, path: Path | None = None):
        self.path = path or runtime_dir() / 'runs.sqlite'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS task_links (task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, body TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS script_mail (
                    run_id TEXT NOT NULL, outcome TEXT NOT NULL, sent_at REAL NOT NULL,
                    PRIMARY KEY(run_id, outcome)
                );
                CREATE INDEX IF NOT EXISTS runs_phase ON runs(json_extract(data, '$.phase'));
            ''')

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=15)

    def put(self, run: dict) -> None:
        with self.connect() as connection:
            connection.execute('INSERT INTO runs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data', (run['id'], json.dumps(run)))

    def get(self, run_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute('SELECT data FROM runs WHERE id=?', (run_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def active(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT data FROM runs WHERE json_extract(data, '$.phase') IN ('starting','running','terminating','finalizing')").fetchall()
        return [json.loads(row[0]) for row in rows]

    def owned(self, owner: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT data FROM runs WHERE json_extract(data, '$.owner')=? ORDER BY rowid", (owner,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def link(self, task_id: str, run_id: str) -> None:
        with self.connect() as connection:
            connection.execute('INSERT OR IGNORE INTO task_links VALUES (?, ?)', (task_id, run_id))

    def links(self, run_id: str) -> list[str]:
        with self.connect() as connection:
            return [row[0] for row in connection.execute('SELECT task_id FROM task_links WHERE run_id=?', (run_id,))]

    def event(self, run: dict, kind: str, detail: str) -> None:
        body = json.dumps({'run': run, 'detail': detail, 'kind': kind})
        with self.connect() as connection:
            connection.execute('INSERT OR IGNORE INTO events(id,body) VALUES (?,?)', (f'{run["id"]}:{kind}', body))


def record_script_email(*, success: bool) -> None:
    """Receipt after SMTP success, separate from concurrently updated run state."""
    run_id = os.getenv('LEARNDL_MANAGED_RUN')
    if not run_id:
        return
    store = RunStore()
    run = store.get(run_id)
    if run.get('owner') != 'idle_worklist' or run.get('pid') != os.getpid():
        return
    with store.connect() as connection:
        connection.execute('INSERT OR REPLACE INTO script_mail VALUES (?, ?, ?)',
                           (run_id, 'success' if success else 'error', time.time()))


def idle_exit_needs_email(store: RunStore, run: dict) -> bool:
    """Keep legacy no-script-mail runs covered across a rolling code update."""
    if run['phase'] == 'deferred':
        return False
    # Signals, reboot, launcher failure and watchdog termination are distinct
    # from an ordinary Python failure which the script may already have mailed.
    code = run.get('returncode')
    if 'term_at' in run or code is None or code < 0:
        return True
    if code == 75:
        return False
    if code == 0 or run['phase'] == 'complete':
        if run.get('script_email_enabled'):
            return False
        # Old workers launched training with email=False. Missing metadata must
        # therefore retain their watchdog result mail unless the script sent it.
        with store.connect() as connection:
            receipt = connection.execute("SELECT 1 FROM script_mail WHERE run_id=? AND outcome='success'",
                                         (run['id'],)).fetchone()
        return receipt is None
    with store.connect() as connection:
        receipt = connection.execute("SELECT 1 FROM script_mail WHERE run_id=? AND outcome='error'",
                                     (run['id'],)).fetchone()
    return receipt is None


def register_task(task_id: str) -> None:
    run_id = os.getenv('LEARNDL_MANAGED_RUN')
    if not run_id:
        return
    store = RunStore()
    run = store.get(run_id)
    if run and os.getsid(0) == run['pid']:
        store.link(task_id, run_id)


def timeout_override(task_id: str) -> dict[str, Any]:
    """A late recorder exit cannot overwrite a confirmed timeout outcome."""
    path = runtime_dir() / 'runs.sqlite'
    if not path.exists():
        return {}
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=15) as connection:
        row = connection.execute('SELECT data FROM runs JOIN task_links ON runs.id=task_links.run_id WHERE task_id=?', (task_id,)).fetchone()
    if row:
        run = json.loads(row[0])
        if run['phase'] in {'killed', 'finalizing'}:
            return {'status': 'killed', 'end_time': run['finished_at'], 'exit_code': -9 if run.get('kill_at') else -15,
                    'exit_error': timeout_message(run)}
    return {}


def process_cgroup(pid: int) -> str | None:
    try:
        for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines():
            hierarchy, controllers, path = line.split(':', 2)
            if hierarchy == '0' or controllers == 'name=systemd':
                return path
    except FileNotFoundError:
        pass
    return None


def termination_requested() -> bool:
    run_id = os.getenv('LEARNDL_MANAGED_RUN')
    return bool(run_id and 'term_at' in RunStore().get(run_id))


def in_run_scope(pid: int, run: dict) -> bool:
    if not run.get('cgroup'):
        return os.getsid(pid) == run['pid']
    group = process_cgroup(pid)
    return group is not None and (group == run['cgroup'] or group.startswith(run['cgroup'] + '/'))


def boot_id() -> str | None:
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except FileNotFoundError:
        return None


def previous_boot(run: dict) -> bool:
    return bool(run.get('boot_id') and run['boot_id'] != boot_id())


def members(run: dict) -> list[psutil.Process]:
    """Verify root identity and discover members of the dedicated session.

    Never use an unverified/reused root PID as a signal target. Descendants in
    the original session can remain after the root exits; PID creation times
    are checked again by psutil before each signal.
    """
    if previous_boot(run):
        return []
    try:
        root = psutil.Process(run['pid'])
        if abs(root.create_time() - run['process_created']) > .01:
            raise RuntimeError('Root PID has been reused; refusing termination')
    except psutil.NoSuchProcess:
        pass
    result = []
    for proc in psutil.process_iter(['pid', 'create_time', 'status', 'uids']):
        try:
            if not in_run_scope(proc.pid, run) or proc.pid == run.get('runner_pid') or proc.status() == psutil.STATUS_ZOMBIE:
                continue
            if proc.uids().real != run['uid'] or proc.create_time() < run['process_created'] - .01:
                raise RuntimeError('Session identity mismatch')
            result.append(proc)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
    return result


def signal_members(run: dict, processes: list[psutil.Process], sig: signal.Signals) -> None:
    # Per-process signalling keeps PID-reuse checks inside psutil. Unlike
    # killpg, a newly reused numeric group is never blindly signalled.
    for proc in reversed(processes):
        try:
            if not in_run_scope(proc.pid, run) or proc.pid == run.get('runner_pid'):
                continue
            proc.send_signal(sig)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue


def timeout_message(run: dict) -> str:
    limit = run.get('progress_timeout_seconds') or run.get('timeout_seconds')
    condition = 'without output' if run.get('progress_timeout_seconds') else 'elapsed'
    return f'TIMEOUT: exceeded {limit} seconds {condition}; run {run["id"]}'


def timeout_due(run: dict, now: float) -> bool:
    if 'term_at' in run:
        return True
    if limit := run.get('progress_timeout_seconds'):
        last = run['started_at']
        if run.get('log_path'):
            try:
                last = max(last, Path(run['log_path']).stat().st_mtime)
            except FileNotFoundError:
                pass
        return now - last >= limit
    limit = run.get('timeout_seconds')
    return limit is not None and now - run['started_at'] >= limit


def finish_idle_event(store: RunStore, run: dict, detail: str) -> None:
    if run.get('owner') != 'idle_worklist':
        return
    if run.get('returncode') == 75 and 'term_at' not in run:
        run['phase'] = 'deferred'
        store.put(run)
    if not idle_exit_needs_email(store, run):
        return
    # Preserve the outcome context even if a later manual run replaces worklist state.
    try:
        from src.api.calls.worklist_state import WorklistState
        state = WorklistState(run['task']).read()
        if state and state.get('revisions') == run.get('revisions'):
            run['completion'] = state
    except (OSError, ValueError):
        pass
    store.event(run, 'finished', detail)
    if not run.get('script_email_enabled'):
        # Recover legacy completion mail suppressed by the brief prior policy;
        # sent=1 is never reset, so delivered notifications remain deduplicated.
        with store.connect() as connection:
            connection.execute('UPDATE events SET sent=0 WHERE id=? AND sent=-1',
                               (f'{run["id"]}:finished',))


def inspect_timeouts(store: RunStore | None = None, *, now: float | None = None) -> dict:
    store = store or RunStore()
    now = time.time() if now is None else now
    checked = 0
    for run in store.active():
        if run['phase'] == 'starting':
            # No child exists until runner has persisted its identity. A
            # crashed launcher before that point cannot be safely signalled.
            if now - run['started_at'] > 60:
                run.update(phase='error', finished_at=now)
                store.put(run)
                finish_idle_event(store, run, 'Launcher exited before registering a child.')
            continue
        checked += 1
        if run.get('owner') == 'idle_worklist':
            store.event(run, 'started', 'Automatic schedule process started.')
        try:
            processes = members(run)
            if not processes:
                if 'returncode' not in run and run.get('runner_pid') and not previous_boot(run):
                    try:
                        runner = psutil.Process(run['runner_pid'])
                        if abs(runner.create_time() - run['runner_created']) < .01 and runner.status() != psutil.STATUS_ZOMBIE:
                            continue  # Let the live runner persist the actual exit code.
                    except psutil.NoSuchProcess:
                        pass
                timed_out = 'term_at' in run
                run.update(phase='finalizing' if timed_out else ('error' if run.get('returncode', 1) else 'complete'), finished_at=run.get('finished_at', now))
                store.put(run)
                if timed_out:
                    from src.api.util.backend.task import TaskDatabase
                    database = TaskDatabase()
                    for task_id in store.links(run['id']):
                        task = database.get_task(task_id)
                        if task is None or (task.pid != run['pid'] and not task.is_running):
                            continue
                        database.update_task(task_id, backend_updated=True, **{
                            'status': 'killed', 'end_time': run['finished_at'], 'exit_code': -9 if run.get('kill_at') else -15,
                            'exit_error': timeout_message(run),
                        })
                    run['phase'] = 'killed'
                    if run.get('owner') != 'idle_worklist':
                        store.event(run, 'terminated', 'Timed out; verified session processes have exited.')
                    store.put(run)
                finish_idle_event(store, run, f'Process finished: {run["phase"]}; returncode={run.get("returncode", "unavailable (abnormal exit)")}.')
                continue
            if not timeout_due(run, now):
                continue
            if 'term_at' not in run:
                run.update(phase='terminating', term_at=now)
                store.put(run)  # Persist intent before sending any signal.
            if now - run['term_at'] < 30:
                signal_members(run, processes, signal.SIGTERM)
            else:
                run['kill_at'] = run.get('kill_at', now)
                store.put(run)
                signal_members(run, processes, signal.SIGKILL)
                if now - run['kill_at'] >= 60:
                    store.event(run, 'failed', 'SIGKILL sent but processes are still present; will retry.')
        except (psutil.Error, OSError, RuntimeError) as exc:
            if timeout_due(run, now):
                store.event(run, 'failed', f'Timeout termination could not be verified/completed: {exc}')
    # Repair event persistence after a crash between storing state and inserting mail.
    for run in store.owned('idle_worklist'):
        if run.get('pid'):
            store.event(dict(run, phase='running'), 'started', 'Automatic schedule process started.')
        if run['phase'] in {'complete', 'error', 'killed', 'deferred'} and run.get('started_at'):
            finish_idle_event(store, run, f'Process finished: {run["phase"]}; returncode={run.get("returncode", "unavailable (abnormal exit)")}.')
    return {'checked_runs': checked}


def deliver_timeout_events(sender: Any, store: RunStore | None = None) -> bool:
    store = store or RunStore()
    with store.connect() as connection:
        rows = connection.execute('SELECT id,body FROM events WHERE sent=0').fetchall()
    success = True
    for event_id, body in rows:
        event = json.loads(body)
        run = event['run']
        if run.get('owner') == 'idle_worklist' and event['kind'] == 'finished':
            # Suppress new-policy success mails and errors reported since
            # this event was created; retain the event for audit (sent=-1).
            current = store.get(run['id']) or run
            if not idle_exit_needs_email(store, current):
                with store.connect() as connection:
                    connection.execute('UPDATE events SET sent=-1 WHERE id=?', (event_id,))
                continue
        from src.api.util.backend.task import TaskDatabase
        database = TaskDatabase()
        attachments = []
        for task_id in store.links(run['id']):
            task = database.get_task(task_id)
            if task:
                attachments.extend(Path(path) for path in task.exit_files or [] if Path(path).is_file())
                attachments.extend(Path(path) for path in task.get_crash_protector() if Path(path).is_file())
        label = 'Idle Worklist' if run.get('owner') == 'idle_worklist' else 'Timeout'
        message = (f'Task: {run["task"]}\nRun: {run["id"]}\nStarted: {run["started_at"]}\n'
                   f'Limit: {run["timeout_seconds"]} seconds\nElapsed: {time.time() - run["started_at"]:.0f} seconds\n'
                   f'Phase: {run["phase"]}\n{event["detail"]}\n'
                   f'Log: {run.get("log_path", "")}\nVersion: {run.get("version", "")}\n'
                   f'Progress timeout: {run.get("progress_timeout_seconds", "")} seconds')
        if run.get('owner') == 'idle_worklist':
            completion = run.get('completion') or {}
            message += (f'\nModel directory: {completion.get("model_path", "pending")}\n'
                        f'Training runs: {completion.get("training_run_ids", [])}\n'
                        f'Error: {completion.get("error", "")}\n')
            for name, revision in run.get('revisions', {}).items():
                message += f'{name} revision: {revision.get("sha256")} git: {revision.get("git_commit")}\n'
        if sender(f'Watchdog {label} - {run["task"]} - {event["kind"]}', message,
                  attachments=list(dict.fromkeys(attachments)), confirmation_message='Learndl timeout alert'):
            with store.connect() as connection:
                connection.execute('UPDATE events SET sent=1 WHERE id=?', (event_id,))
        else:
            success = False
    return success


def timeout_task_ids() -> set[str]:
    """Exclude timeout events from the generic killed notification path."""
    path = runtime_dir() / 'runs.sqlite'
    if not path.exists():
        return set()
    store = RunStore(path)
    with store.connect() as connection:
        rows = connection.execute('SELECT task_id,data FROM task_links JOIN runs ON runs.id=task_links.run_id').fetchall()
    return {task_id for task_id, data in rows if 'term_at' in json.loads(data) or json.loads(data).get('owner') == 'idle_worklist'}
