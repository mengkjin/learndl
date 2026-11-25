"""Explicit operator actions, separate from the monitor's read-only queries."""
from __future__ import annotations

__all__ = [
    'ALIVE_STATUSES',
    'KillTarget',
    'KillResult',
    'kill_unavailable_reason',
    'prepare_kill',
    'kill_task',
]

import os
from dataclasses import dataclass
from typing import Literal

import psutil

from src.api.task_monitor.core import TaskSnapshot
from src.api.util.backend.task import TaskDatabase, timestamp

ALIVE_STATUSES = frozenset({'running', 'sleeping', 'disk-sleep'})


@dataclass(frozen=True)
class KillTarget:
    task_id: str
    pid: int
    process_created: float
    script: str
    started: float
    managed_identity: dict | None = None


@dataclass(frozen=True)
class KillResult:
    status: Literal['success', 'invalid', 'failed']
    message: str


def kill_unavailable_reason(task: TaskSnapshot) -> str | None:
    if task.managed_kind is not None and task.managed_kind != 'scheduler':
        return 'This managed task type requires its scheduler stop control.'
    if not task.is_active:
        return 'This task has finished.'
    if not task.pid or task.pid <= 0:
        return 'No process PID is available yet.'
    if task.pid == os.getpid():
        return 'The Monitor cannot terminate itself.'
    try:
        if psutil.Process(task.pid).status() not in ALIVE_STATUSES:
            return 'This process is no longer running.'
    except psutil.NoSuchProcess:
        return 'This process has exited.'
    except psutil.Error as exc:
        return f'Cannot inspect this process: {exc}'
    return None


def prepare_kill(task: TaskSnapshot) -> KillTarget:
    """Capture an immutable identity before asking the operator to confirm."""
    if reason := kill_unavailable_reason(task):
        raise ValueError(reason)
    assert task.pid is not None
    if task.managed_kind == 'scheduler':
        from src.automation.scheduling.operator_stop import prepare
        created = task.process_created
        if created is None:
            raise ValueError('Missing managed process creation time.')
        identity = prepare(task.task_id, task.pid, created)
        return KillTarget(task.task_id, task.pid, created, task.script, task.effective_start, identity)
    return KillTarget(task.task_id, task.pid, psutil.Process(task.pid).create_time(),
                      task.script, task.effective_start)


def kill_task(target: KillTarget) -> KillResult:
    """Revalidate the confirmed identity and queue mail for a verified stop."""
    if target.pid <= 0 or target.pid == os.getpid():
        return KillResult('invalid', 'Invalid target or Monitor process; kill skipped.')
    try:
        if target.managed_identity is not None:
            from src.automation.scheduling.operator_stop import stop
            return KillResult('success',stop(target.task_id,target.managed_identity))
        if target.task_id.startswith('managed:'):
            return KillResult('invalid','Managed task requires a confirmed group identity.')
        db = TaskDatabase()
        current = db.get_task(target.task_id)
        if current is None or current.id != target.task_id or not current.is_running or current.pid != target.pid:
            return KillResult('invalid', 'The task has finished or its PID changed; kill skipped.')
        proc = psutil.Process(target.pid)
        if proc.create_time() != target.process_created or proc.status() not in ALIVE_STATUSES:
            return KillResult('invalid', 'The process identity or status changed; kill skipped.')
        from src.automation.scheduling.runtime import OPERATOR_STOP_EVENT, OPERATOR_TASK_PREFIX, RunStore, pin_crash_logs

        store = RunStore()
        run_id = f'{OPERATOR_TASK_PREFIX}{target.task_id}'
        paths = pin_crash_logs(store, current)
        logs = {target.task_id: paths}
        detail = 'Killed by operator from Learndl Monitor'
        run = dict(id=run_id, task=target.script, pid=target.pid, process_created=target.process_created,
                   stop_requested_at=timestamp(), operator_crash_logs=logs)
        if not store.event(run, OPERATOR_STOP_EVENT, detail, awaiting_exit=True):
            return KillResult('invalid', 'A stop has already been requested for this task.')
        finished_at = None
        try:
            proc.terminate()
            try:
                exit_code = proc.wait(timeout=2)
            except psutil.TimeoutExpired:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
                exit_code = proc.wait(timeout=2)
            finished_at = timestamp()
            # The recorder may have written final output during SIGTERM cleanup.
            latest = db.get_task(target.task_id)
            if latest and latest.pid == target.pid:
                paths = list(dict.fromkeys([*paths, *pin_crash_logs(store, latest)]))
                logs[target.task_id] = paths
                db.update_task(target.task_id, backend_updated=True, status='killed', end_time=finished_at,
                               exit_code=exit_code if exit_code is not None else 1, exit_error=detail,
                               exit_files=list(dict.fromkeys([*paths, *(latest.exit_files or [])])))
        finally:
            store.finish_operator_task(run_id, finished_at, logs)
        return KillResult('success', f'Killed PID {target.pid} ({target.script}).')
    except psutil.NoSuchProcess:
        return KillResult('invalid', 'The process has exited; kill skipped.')
    except Exception as exc:
        return KillResult('failed', f'Unable to finish kill operation: {exc}')
