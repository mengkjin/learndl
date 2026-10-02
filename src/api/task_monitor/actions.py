"""Explicit operator actions, separate from the monitor's read-only queries."""
from __future__ import annotations

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


@dataclass(frozen=True)
class KillResult:
    status: Literal['success', 'invalid', 'failed']
    message: str


def kill_unavailable_reason(task: TaskSnapshot) -> str | None:
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
    return KillTarget(task.task_id, task.pid, psutil.Process(task.pid).create_time(),
                      task.script, task.effective_start)


def kill_task(target: KillTarget) -> KillResult:
    """Revalidate the confirmed identity and use the same lifecycle operation as CLI."""
    if target.pid <= 0 or target.pid == os.getpid():
        return KillResult('invalid', 'Invalid target or Monitor process; kill skipped.')
    try:
        db = TaskDatabase()
        current = db.get_task(target.task_id)
        if current is None or current.id != target.task_id or not current.is_running or current.pid != target.pid:
            return KillResult('invalid', 'The task has finished or its PID changed; kill skipped.')
        current.set_task_db(db)
        proc = psutil.Process(target.pid)
        if proc.create_time() != target.process_created or proc.status() not in ALIVE_STATUSES:
            return KillResult('invalid', 'The process identity or status changed; kill skipped.')
        if not current.kill():
            return KillResult('failed', f'Failed to kill PID {target.pid}.')
        if current.is_running:
            current.update({
                'status': 'killed', 'end_time': timestamp(), 'exit_code': 1,
                'exit_error': 'Killed by operator from Learndl Monitor',
            }, sync=True)
        return KillResult('success', f'Killed PID {target.pid} ({target.script}).')
    except psutil.NoSuchProcess:
        return KillResult('invalid', 'The process has exited; kill skipped.')
    except Exception as exc:
        return KillResult('failed', f'Unable to finish kill operation: {exc}')
