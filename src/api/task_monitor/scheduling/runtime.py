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
                    'exit_error': f'TIMEOUT: exceeded {run["timeout_seconds"]} seconds; run {run["id"]}'}
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


def members(run: dict) -> list[psutil.Process]:
    """Verify root identity and discover members of the dedicated session.

    Never use an unverified/reused root PID as a signal target. Descendants in
    the original session can remain after the root exits; PID creation times
    are checked again by psutil before each signal.
    """
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
            continue
        checked += 1
        try:
            processes = members(run)
            if not processes:
                if 'returncode' not in run and run.get('runner_pid'):
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
                            'exit_error': f'TIMEOUT: exceeded {run["timeout_seconds"]} seconds; run {run["id"]}',
                        })
                    run['phase'] = 'killed'
                    store.event(run, 'terminated', 'Timed out; verified session processes have exited.')
                    store.put(run)
                continue
            limit = run['timeout_seconds']
            if limit is None or now - run['started_at'] < limit:
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
            if run.get('timeout_seconds') is not None and now - run['started_at'] >= run['timeout_seconds']:
                store.event(run, 'failed', f'Timeout termination could not be verified/completed: {exc}')
    return {'checked_runs': checked}


def deliver_timeout_events(sender: Any, store: RunStore | None = None) -> bool:
    store = store or RunStore()
    with store.connect() as connection:
        rows = connection.execute('SELECT id,body FROM events WHERE sent=0').fetchall()
    success = True
    for event_id, body in rows:
        event = json.loads(body)
        run = event['run']
        from src.api.util.backend.task import TaskDatabase
        database = TaskDatabase()
        attachments = []
        for task_id in store.links(run['id']):
            task = database.get_task(task_id)
            if task:
                attachments.extend(Path(path) for path in task.exit_files or [] if Path(path).is_file())
                attachments.extend(Path(path) for path in task.get_crash_protector() if Path(path).is_file())
        message = (f'Task: {run["task"]}\nRun: {run["id"]}\nStarted: {run["started_at"]}\n'
                   f'Limit: {run["timeout_seconds"]} seconds\nElapsed: {time.time() - run["started_at"]:.0f} seconds\n'
                   f'Phase: {run["phase"]}\n{event["detail"]}')
        if sender(f'Watchdog Timeout - {run["task"]} - {event["kind"]}', message,
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
    return {task_id for task_id, data in rows if 'term_at' in json.loads(data)}
