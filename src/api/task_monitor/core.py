"""Read-only task queries and lightweight monitor metadata helpers."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import psutil

from src.proj import PATH
from src.api.task_monitor.output import parse_live_output
from src.api.util.backend.task import TaskDatabase

TaskStatus: TypeAlias = Literal['running', 'complete', 'error', 'killed']
TimeWindow: TypeAlias = Literal['history', 'last_day', 'last_week', 'last_month']
PageCursor: TypeAlias = tuple[float, str]

ACTIVE_STATUSES = frozenset(('starting', 'running'))
FINISHED_STATUSES = frozenset(('complete', 'error', 'killed'))
BACKGROUND_SOURCES = frozenset(('bash', 'cron', 'nohup', 'systemd'))
TIME_WINDOW_SECONDS: dict[TimeWindow, float | None] = {
    'history': None,
    'last_day': 86400,
    'last_week': 7 * 86400,
    'last_month': 30 * 86400,
}

__all__ = [
    'ACTIVE_STATUSES', 'FINISHED_STATUSES', 'PageCursor', 'WatchdogHealth',
    'TaskMonitorRepository', 'TaskPage', 'TaskSnapshot', 'TaskStatus', 'TimeWindow',
    'is_background_source', 'read_live_log',
]


@dataclass(frozen=True)
class TaskSnapshot:
    """One task record, optionally enriched with selected-detail metadata."""

    task_id: str
    script: str
    cmd: str
    create_time: float
    status: str
    source: str | None
    pid: int | None
    start_time: float | None
    end_time: float | None
    exit_code: int | None
    exit_message: str | None
    exit_error: str | None
    exit_files: tuple[Path, ...] = ()
    crash_log: Path | None = None
    pid_alive: bool | None = None

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def is_background(self) -> bool:
        return is_background_source(self.source)

    @property
    def effective_start(self) -> float:
        return self.start_time or self.create_time

    @property
    def display_status(self) -> str:
        if self.is_active and self.pid is not None and self.pid_alive is False:
            return 'stale'
        return self.status


@dataclass(frozen=True)
class TaskPage:
    tasks: tuple[TaskSnapshot, ...]
    total: int
    next_cursor: PageCursor | None
    has_active: bool


@dataclass(frozen=True)
class WatchdogHealth:
    last_run: float | None
    stale: bool
    payload: dict[str, object]


def is_background_source(source: str | None) -> bool:
    return (source or '').strip().lower() in BACKGROUND_SOURCES


def _pid_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        return psutil.pid_exists(pid)
    except psutil.Error:
        return None


def _crash_log_path(task_id: str, runtime_dir: Path) -> Path | None:
    suffix = f'.{task_id.replace("/", "_")}.md'
    crash_dir = runtime_dir / 'crash_protector'
    if not crash_dir.is_dir():
        return None
    try:
        matches = sorted(crash_dir.glob(f'*{suffix}'), key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return None
    return matches[0] if matches else None


def read_live_log(path: Path | None, *, max_bytes: int = 512 * 1024, max_lines: int = 500) -> str:
    """Backward-compatible plain-text live tail."""
    return parse_live_output(path, 'text', max_bytes=max_bytes, max_lines=max_lines)


class TaskMonitorRepository:
    """Read task metadata without mutating lifecycle state."""

    def __init__(self, db_path: Path | None = None, runtime_dir: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else TaskDatabase.get_db_path()
        self.runtime_dir = Path(runtime_dir) if runtime_dir is not None else PATH.runtime

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f'{self.db_path.as_uri()}?mode=ro', uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only = ON')
        return connection

    @staticmethod
    def _database_statuses(statuses: set[TaskStatus]) -> tuple[str, ...]:
        result: list[str] = []
        for status in ('running', 'complete', 'error', 'killed'):
            if status not in statuses:
                continue
            result.extend(('starting', 'running') if status == 'running' else (status,))
        return tuple(result)

    @staticmethod
    def _cutoff(window: TimeWindow, now: float | None) -> float | None:
        seconds = TIME_WINDOW_SECONDS[window]
        return None if seconds is None else (time.time() if now is None else now) - seconds

    @staticmethod
    def _snapshot(
        record: sqlite3.Row, *, detail: bool = False, files: tuple[Path, ...] = (),
        runtime_dir: Path | None = None,
    ) -> TaskSnapshot:
        status = record['status']
        pid = record['pid']
        return TaskSnapshot(
            task_id=record['task_id'], script=record['script'], cmd=record['cmd'],
            create_time=record['create_time'], status=status, source=record['source'], pid=pid,
            start_time=record['start_time'], end_time=record['end_time'], exit_code=record['exit_code'],
            exit_message=record['exit_message'], exit_error=record['exit_error'], exit_files=files,
            crash_log=_crash_log_path(record['task_id'], runtime_dir) if detail and runtime_dir is not None else None,
            pid_alive=_pid_alive(pid) if status in ACTIVE_STATUSES else None,
        )

    def list_tasks(
        self, *, statuses: set[TaskStatus] | None = None, window: TimeWindow = 'last_day',
        cursor: PageCursor | None = None, limit: int = 50, now: float | None = None,
    ) -> TaskPage:
        """Return one metadata-only page using a stable descending cursor."""
        if not self.db_path.is_file():
            return TaskPage((), 0, None, False)
        selected: set[TaskStatus] = (
            {'running', 'complete', 'error', 'killed'}
            if statuses is None
            else statuses
        )
        database_statuses = self._database_statuses(selected)
        if not database_statuses:
            return TaskPage((), 0, None, False)
        where = [f"status IN ({','.join('?' for _ in database_statuses)})"]
        params: list[object] = list(database_statuses)
        if (cutoff := self._cutoff(window, now)) is not None:
            where.append('COALESCE(start_time, create_time) >= ?')
            params.append(cutoff)
        count_where = list(where)
        count_params = list(params)
        if cursor is not None:
            where.append(
                '(COALESCE(start_time, create_time) < ? OR '
                '(COALESCE(start_time, create_time) = ? AND task_id < ?))'
            )
            params.extend((cursor[0], cursor[0], cursor[1]))
        where_sql = ' AND '.join(where)
        query = f"""
            SELECT task_id, script, cmd, create_time, status, source, pid, start_time,
                   end_time, exit_code, exit_message, exit_error
            FROM task_records
            WHERE {where_sql}
            ORDER BY COALESCE(start_time, create_time) DESC, task_id DESC
            LIMIT ?
        """
        count_query = f"SELECT COUNT(*) FROM task_records WHERE {' AND '.join(count_where)}"
        try:
            with self._connect() as connection:
                total = int(connection.execute(count_query, count_params).fetchone()[0])
                records = connection.execute(query, (*params, limit + 1)).fetchall()
        except sqlite3.Error:
            return TaskPage((), 0, None, False)
        has_more = len(records) > limit
        records = records[:limit]
        tasks = tuple(self._snapshot(record) for record in records)
        next_cursor = None
        if has_more and tasks:
            last = tasks[-1]
            next_cursor = (last.effective_start, last.task_id)
        return TaskPage(tasks, total, next_cursor, any(task.is_active for task in tasks))

    def get_task(self, task_id: str) -> TaskSnapshot | None:
        """Load exit files and crash metadata for one selected task only."""
        if not self.db_path.is_file():
            return None
        try:
            with self._connect() as connection:
                record = connection.execute('SELECT * FROM task_records WHERE task_id = ?', (task_id,)).fetchone()
                if record is None:
                    return None
                files = tuple(
                    Path(row['file_path']) for row in connection.execute(
                        'SELECT file_path FROM task_exit_files WHERE task_id = ? ORDER BY id', (task_id,),
                    ).fetchall()
                )
        except sqlite3.Error:
            return None
        return self._snapshot(record, detail=True, files=files, runtime_dir=self.runtime_dir)

    def status_counts(
        self, *, statuses: set[TaskStatus] | None = None, window: TimeWindow = 'last_day',
        now: float | None = None,
    ) -> dict[str, int]:
        """Count logical monitor statuses without loading task rows."""
        selected: set[TaskStatus] = (
            {'running', 'complete', 'error', 'killed'}
            if statuses is None
            else statuses
        )
        database_statuses = self._database_statuses(selected)
        counts = {status: 0 for status in ('running', 'complete', 'error', 'killed')}
        if not self.db_path.is_file() or not database_statuses:
            return counts
        where = [f"status IN ({','.join('?' for _ in database_statuses)})"]
        params: list[object] = list(database_statuses)
        if (cutoff := self._cutoff(window, now)) is not None:
            where.append('COALESCE(start_time, create_time) >= ?')
            params.append(cutoff)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"SELECT status, COUNT(*) AS count FROM task_records WHERE {' AND '.join(where)} GROUP BY status",
                    params,
                ).fetchall()
        except sqlite3.Error:
            return counts
        for row in rows:
            logical = 'running' if row['status'] in ACTIVE_STATUSES else row['status']
            counts[logical] += int(row['count'])
        return counts

    def watchdog_health(
        self, job_name: str, stale_after_seconds: float,
    ) -> WatchdogHealth:
        """Read one watchdog job heartbeat without mutating the task database."""
        heartbeat = self.runtime_dir / 'task_watchdog' / 'state.json'
        try:
            payload = json.loads(heartbeat.read_text(encoding='utf-8'))
            job = payload['jobs'][job_name]
            last_run = float(job['last_success_at'])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return WatchdogHealth(None, True, {})
        return WatchdogHealth(last_run, time.time() - last_run > stale_after_seconds, job)
