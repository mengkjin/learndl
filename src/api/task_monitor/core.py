"""Read-only accessors and presentation helpers for recorded script executions."""

from __future__ import annotations

import html
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from src.proj import PATH
from src.api.util.backend.task import TaskDatabase

__all__ = [
    'TaskMonitorRepository', 'TaskSnapshot', 'read_live_log', 'is_background_source',
]

ACTIVE_STATUSES = frozenset(('starting', 'running'))
FINISHED_STATUSES = frozenset(('complete', 'error', 'killed'))
BACKGROUND_SOURCES = frozenset(('bash', 'cron', 'nohup', 'systemd'))
_HTML_TAG = re.compile(r'</?(?:span|u)[^>]*>', flags=re.IGNORECASE)
_LOG_PREFIX = re.compile(r'^- \d{2}:\d{2}:\d{2}\.\d{3}:\s*')
_CONTROL_ONLY = re.compile(r'^\^+$')


@dataclass(frozen=True)
class TaskSnapshot:
    """A task record enriched with read-only process and output metadata."""

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
    exit_files: tuple[Path, ...]
    crash_log: Path | None
    pid_alive: bool | None

    @property
    def is_active(self) -> bool:
        """Whether the backend still represents the task as active."""
        return self.status in ACTIVE_STATUSES

    @property
    def is_background(self) -> bool:
        """Whether the recorded source denotes a detached/background launch."""
        return is_background_source(self.source)

    @property
    def display_status(self) -> str:
        """Return a non-mutating status suitable for the monitor."""
        if self.is_active and self.pid is not None and self.pid_alive is False:
            return 'stale'
        return self.status


def is_background_source(source: str | None) -> bool:
    """Recognise sources conventionally used for unattended task execution."""
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
    matches = sorted(crash_dir.glob(f'*{suffix}'), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _compact_log_lines(lines: list[str], max_lines: int) -> list[str]:
    """Remove empty lines and collapse consecutive terminal-control noise."""
    compacted: list[str] = []
    suppressed = 0
    for line in lines:
        payload = _LOG_PREFIX.sub('', line).strip()
        if not payload or _CONTROL_ONLY.fullmatch(payload):
            suppressed += 1
            continue
        if suppressed:
            compacted.append(f'… {suppressed} blank/control-output lines suppressed …')
            suppressed = 0
        compacted.append(line)
    if suppressed:
        compacted.append(f'… {suppressed} blank/control-output lines suppressed …')
    return compacted[-max_lines:]


def read_live_log(path: Path | None, *, max_bytes: int = 256 * 1024, max_lines: int = 200) -> str:
    """Read a bounded, plain-text tail of a crash-protector markdown log.

    The crash protector deliberately includes a small set of HTML tags for
    formatting.  The monitor renders plain text, so no task output can execute
    in the browser.
    """
    if path is None or not path.is_file():
        return ''
    try:
        with path.open('rb') as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_bytes))
            raw = file.read()
    except OSError:
        return ''
    text = raw.decode('utf-8', errors='replace')
    if len(raw) == max_bytes:
        text = text.split('\n', 1)[-1]
    text = html.unescape(_HTML_TAG.sub('', text))
    # Read extra source lines before compacting: a long tail of terminal noise
    # must not crowd the useful traceback out of the visible window.
    return '\n'.join(_compact_log_lines(text.splitlines()[-max_lines * 5:], max_lines))


class TaskMonitorRepository:
    """Fetch monitor data without updating task state, files, or notifications."""

    def __init__(self, db_path: Path | None = None, runtime_dir: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else TaskDatabase.get_db_path()
        self.runtime_dir = Path(runtime_dir) if runtime_dir is not None else PATH.runtime

    def reconcile_stopped_tasks(self) -> None:
        """Finalize dead recorded processes before displaying their status.

        Only the canonical project database is reconciled.  Custom paths are
        used by tests and diagnostic callers and remain read-only.
        """
        if self.db_path == TaskDatabase.get_db_path():
            task_db = TaskDatabase()
            task_db.reconcile_stopped_tasks()
            task_db.prune_recovered_crash_logs()

    def list_tasks(self, *, now: float | None = None, lookback_hours: float = 24, limit: int = 200) -> list[TaskSnapshot]:
        """Return active tasks plus finished tasks ended within ``lookback_hours``."""
        self.reconcile_stopped_tasks()
        if not self.db_path.is_file():
            return []
        cutoff = (time.time() if now is None else now) - lookback_hours * 3600
        query = """
            SELECT task_id, script, cmd, create_time, status, source, pid, start_time,
                   end_time, exit_code, exit_message, exit_error
            FROM task_records
            WHERE status IN ('starting', 'running')
               OR (status IN ('complete', 'error', 'killed') AND end_time >= ?)
            ORDER BY CASE WHEN status IN ('starting', 'running') THEN 0 ELSE 1 END,
                     COALESCE(start_time, create_time) DESC
            LIMIT ?
        """
        try:
            connection = sqlite3.connect(f'{self.db_path.as_uri()}?mode=ro', uri=True)
            connection.row_factory = sqlite3.Row
            with connection:
                records = connection.execute(query, (cutoff, limit)).fetchall()
                files = connection.execute(
                    'SELECT task_id, file_path FROM task_exit_files WHERE task_id IN ('
                    + ','.join('?' for _ in records) + ')',
                    [record['task_id'] for record in records],
                ).fetchall() if records else []
        except sqlite3.Error:
            return []
        finally:
            if 'connection' in locals():
                connection.close()

        files_by_task: dict[str, list[Path]] = {}
        for file_record in files:
            files_by_task.setdefault(file_record['task_id'], []).append(Path(file_record['file_path']))
        return [
            TaskSnapshot(
                task_id=record['task_id'], script=record['script'], cmd=record['cmd'],
                create_time=record['create_time'], status=record['status'], source=record['source'],
                pid=record['pid'], start_time=record['start_time'], end_time=record['end_time'],
                exit_code=record['exit_code'], exit_message=record['exit_message'], exit_error=record['exit_error'],
                exit_files=tuple(files_by_task.get(record['task_id'], [])),
                crash_log=_crash_log_path(record['task_id'], self.runtime_dir),
                pid_alive=_pid_alive(record['pid']),
            )
            for record in records
        ]
