"""System watchdog and extensible short-maintenance runner.

The systemd timer invokes this module once per minute. Individual jobs keep
their own intervals in the durable watchdog state, so adding a small periodic
maintenance operation does not require another cron entry, shell script, or
systemd unit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from src.api.task_monitor.output import OutputCache
from src.api.util.backend.task import TaskDatabase, TaskItem
from src.proj import MACHINE, PATH, Logger
from src.proj.util.script.script_lock import ScriptLock

__all__ = ['WatchdogJob', 'run_watchdog']

STATE_VERSION = 2
TASK_LIFECYCLE_INTERVAL = 60
UNIT_PROBE_INTERVAL = 60
MONITOR_CACHE_INTERVAL = 15 * 60
MAX_CACHE_SOURCES_PER_RUN = 20
MAX_CACHE_SECONDS_PER_RUN = 20
_UNIT_NAME = re.compile(r'^[A-Za-z0-9_.@:-]+$')


class _TaskRepository(Protocol):
    def reconcile_stopped_tasks(self) -> dict[str, dict[str, Any]]: ...
    def get_task(self, task_id: str) -> TaskItem | None: ...
    def get_killed_tasks_since(self, since: float) -> list[TaskItem]: ...
    def recovered_crash_logs_due(self) -> list[tuple[str, Path]]: ...
    def prune_recovered_crash_logs(self, *, cached_paths: set[Path]) -> list[Path]: ...


@dataclass(frozen=True)
class JobResult:
    """Bounded, JSON-safe result returned by one watchdog job."""

    stats: dict[str, object] = field(default_factory=dict)
    pending_tasks: frozenset[str] = frozenset()
    unit_failures: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class WatchdogJob:
    """A small idempotent task scheduled by the one-minute watchdog runner."""

    name: str
    interval_seconds: int
    runner: Callable[['_WatchdogContext'], JobResult]


@dataclass
class _WatchdogContext:
    task_db: _TaskRepository
    cache: OutputCache
    units: Sequence[str]
    previous_check: float
    unit_checker: Callable[[str], tuple[bool, str]]
    maintenance: dict[str, Any] = field(default_factory=dict)


def _default_state_path() -> Path:
    return PATH.runtime / 'task_watchdog' / 'state.json'


def _load_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        value = {}
    state = value if isinstance(value, dict) else {}
    alerts = state.get('alerts')
    if not isinstance(alerts, dict):
        alerts = {
            'pending_tasks': state.pop('pending_tasks', []),
            'unit_alerted': state.pop('unit_alerted', {}),
            'last_check_unix': state.pop('last_check_unix', None),
        }
    alerts.setdefault('pending_tasks', [])
    alerts.setdefault('unit_alerted', {})
    state['alerts'] = alerts
    state['jobs'] = state.get('jobs') if isinstance(state.get('jobs'), dict) else {}
    state['version'] = STATE_VERSION
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _unit_names(values: Sequence[str] | None = None) -> list[str]:
    if values is None:
        values = re.split(r'[\s,]+', os.getenv('LEARNDL_WATCHDOG_UNITS', ''))
    names = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    invalid = [name for name in names if not _UNIT_NAME.fullmatch(name)]
    if invalid:
        raise ValueError(f'Invalid systemd unit name(s): {invalid}')
    return names


def _check_unit(unit: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', unit], capture_output=True, text=True,
            timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f'check failed: {exc!r}'
    status = result.stdout.strip() or result.stderr.strip() or f'exit {result.returncode}'
    return result.returncode == 0 and status == 'active', status


def _run_task_lifecycle(context: _WatchdogContext) -> JobResult:
    """Reconcile stale records and alert only on unexpected termination.

    ``BackendTaskRecorder`` owns normal Python exception handling: it records
    those runs as ``error`` and the script's existing error-email path reports
    them.  Including ``error`` here would turn one application failure into a
    second watchdog email.  The watchdog is therefore deliberately limited to
    lifecycle loss (``killed``), where no orderly recorder completion occurred.
    """
    changed = context.task_db.reconcile_stopped_tasks()
    pending = {
        task_id for task_id, update in changed.items()
        if update.get('status') == 'killed'
    }
    pending.update(task.id for task in context.task_db.get_killed_tasks_since(context.previous_check))
    from src.api.task_monitor.scheduling.runtime import timeout_task_ids
    pending.difference_update(timeout_task_ids())
    return JobResult(
        stats={'reconciled_count': len(changed), 'reconciled_task_ids': sorted(changed)},
        pending_tasks=frozenset(pending),
    )


def _run_unit_probe(context: _WatchdogContext) -> JobResult:
    failures: list[tuple[str, str]] = []
    for unit in context.units:
        active, status = context.unit_checker(unit)
        if not active:
            failures.append((unit, status))
    return JobResult(stats={'checked_units': len(context.units)}, unit_failures=tuple(failures))


def _run_task_monitor_cache(context: _WatchdogContext) -> JobResult:
    due = context.task_db.recovered_crash_logs_due()
    started_at = time.monotonic()
    cached_paths: set[Path] = set()
    failures: list[str] = []
    handled = 0
    for task_id, path in due:
        options = context.maintenance.get('jobs', {}).get('task_monitor_cache', {})
        if handled >= options.get('max_sources', MAX_CACHE_SOURCES_PER_RUN) or time.monotonic() - started_at >= options.get('max_seconds', MAX_CACHE_SECONDS_PER_RUN):
            break
        handled += 1
        try:
            if context.cache.ensure_both(task_id, path):
                cached_paths.add(path)
            else:
                failures.append(str(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f'{path}: {exc}')
    removed = context.task_db.prune_recovered_crash_logs(cached_paths=cached_paths)
    cleanup = context.cache.cleanup()
    return JobResult(stats={
        'due_crash_logs': len(due),
        'handled_crash_logs': handled,
        'deferred_crash_logs': max(0, len(due) - handled),
        'prewarmed_crash_logs': len(cached_paths),
        'removed_crash_logs': len(removed),
        'cache_failures': failures,
        'cache_cleanup': cleanup,
    })


def _run_task_timeouts(context: _WatchdogContext) -> JobResult:
    from src.api.task_monitor.scheduling.runtime import inspect_timeouts
    return JobResult(stats=inspect_timeouts())


def _run_idle_worklist(context: _WatchdogContext) -> JobResult:
    from src.api.task_monitor.scheduling.idle_worklist import dispatch
    return JobResult(stats=dispatch(context.maintenance['jobs']['idle_worklist']))


def _run_git_auto_update(context: _WatchdogContext) -> JobResult:
    from src.api.task_monitor.scheduling.git_update import check_and_update
    return JobResult(stats=check_and_update(context.maintenance['jobs']['git_auto_update']))


def default_jobs(maintenance: dict | None = None) -> tuple[WatchdogJob, ...]:
    """Return the registry; future short maintenance belongs here."""
    defaults = (
        WatchdogJob('task_timeouts', TASK_LIFECYCLE_INTERVAL, _run_task_timeouts),
        WatchdogJob('task_lifecycle', TASK_LIFECYCLE_INTERVAL, _run_task_lifecycle),
        WatchdogJob('systemd_unit_probe', UNIT_PROBE_INTERVAL, _run_unit_probe),
        WatchdogJob('task_monitor_cache', MONITOR_CACHE_INTERVAL, _run_task_monitor_cache),
    )
    if maintenance is None:
        return defaults
    registry = {job.name: job.runner for job in defaults}
    registry['idle_worklist'] = _run_idle_worklist
    registry['git_auto_update'] = _run_git_auto_update
    return tuple(WatchdogJob(name, maintenance['jobs'][name]['interval_seconds'], registry[name])
                 for name in registry if maintenance['jobs'].get(name, {}).get('enabled', False))


def _job_due(job_state: dict[str, Any], interval_seconds: int, now: float) -> bool:
    try:
        last_success = float(job_state.get('last_attempt_completed_at', job_state.get('last_success_at', 0)))
    except (TypeError, ValueError):
        return True
    return now - last_success >= interval_seconds


def _record_job(
    state: dict[str, Any], job: WatchdogJob, *, started_at: float, result: JobResult | None,
    finished_at: float, error: Exception | None = None,
) -> None:
    job_state = state['jobs'].setdefault(job.name, {})
    job_state.update({
        'last_started_at': started_at,
        'last_finished_at': finished_at,
        'duration_seconds': max(0.0, finished_at - started_at),
    })
    if error is None and result is not None:
        job_state.update({'last_success_at': finished_at, 'last_error': None, 'stats': result.stats})
    elif error is not None:
        job_state['last_error'] = f'{type(error).__name__}: {error}'


def _task_description(task_id: str, task: TaskItem | None) -> tuple[str, list[Path]]:
    if task is None:
        return f'Task {task_id}: record unavailable', []
    lines = [
        f'Task: {task_id}',
        f'Script: {task.script}',
        f'Command: {task.cmd}',
        f'PID: {task.pid}',
        f'Status: {task.status}',
        f'Started: {datetime.fromtimestamp(task.start_time).astimezone().isoformat() if task.start_time else "unknown"}',
        f'Error: {task.exit_error or "none recorded"}',
    ]
    attachments = [Path(path) for path in task.exit_files or [] if Path(path).is_file()]
    return '\n'.join(lines), attachments


def _deliver_alerts(
    *, task_db: _TaskRepository, state: dict[str, Any], unit_failures: Sequence[tuple[str, str]],
    state_path: Path, email_sender: Callable[..., bool] | None,
) -> bool:
    alerts = state['alerts']
    pending_tasks = set(str(value) for value in alerts.get('pending_tasks', []))
    from src.api.task_monitor.scheduling.runtime import timeout_task_ids
    pending_tasks.difference_update(timeout_task_ids())
    already_alerted = set(alerts.get('task_alerted', []))
    pending_tasks.difference_update(already_alerted)
    pending_tasks = {task_id for task_id in pending_tasks
                     if (task := task_db.get_task(task_id)) is None or task.status == 'killed'}
    unit_alerted = {str(key): bool(value) for key, value in alerts.get('unit_alerted', {}).items()}
    for unit, _ in unit_failures:
        unit_alerted.setdefault(unit, False)
    new_unit_failures = [(unit, status) for unit, status in unit_failures if not unit_alerted.get(unit, False)]
    alerts.update({'pending_tasks': sorted(pending_tasks), 'unit_alerted': unit_alerted})
    _save_state(state_path, state)
    if not pending_tasks and not new_unit_failures:
        return True

    sections: list[str] = []
    attachments: list[Path] = []
    for task_id in sorted(pending_tasks):
        description, task_attachments = _task_description(task_id, task_db.get_task(task_id))
        sections.append(description)
        attachments.extend(task_attachments)
    for unit, status in new_unit_failures:
        sections.append(f'Systemd unit: {unit}\nStatus: {status}')
    body = '\n\n'.join([
        'The Learndl watchdog detected services or scripts that stopped unexpectedly.',
        *sections,
        f'Watchdog host: {MACHINE.name}',
        f'Watchdog state: {state_path}',
    ])
    if email_sender is None:
        from src.proj.util.web.emailer import Email
        email_sender = Email.send
    sent = email_sender(
        f'Watchdog Alert - {len(pending_tasks) + len(new_unit_failures)} Learndl failure(s)',
        body, attachments=list(dict.fromkeys(attachments)),
        confirmation_message='Learndl watchdog alert',
    )
    if sent:
        for unit, _ in new_unit_failures:
            unit_alerted[unit] = True
        alerts.update({'pending_tasks': [], 'unit_alerted': unit_alerted, 'task_alerted': sorted(already_alerted | pending_tasks)})
        _save_state(state_path, state)
        Logger.success('Learndl watchdog alert delivered')
    else:
        Logger.error('Learndl watchdog alert delivery failed; it will be retried')
    return sent


def run_watchdog(
    *, task_db: _TaskRepository | None = None, cache: OutputCache | None = None,
    units: Sequence[str] | None = None, state_path: Path | None = None,
    email_sender: Callable[..., bool] | None = None,
    unit_checker: Callable[[str], tuple[bool, str]] = _check_unit,
    jobs: Sequence[WatchdogJob] | None = None, now: float | None = None,
) -> bool:
    """Run due watchdog jobs, persist health, then deliver aggregated alerts."""
    task_db = task_db or TaskDatabase()
    cache = cache or OutputCache()
    state_path = state_path or _default_state_path()
    state = _load_state(state_path)
    check_time = time.time() if now is None else now
    alerts = state['alerts']
    from src.api.task_monitor.scheduling.config import installed_config
    config = installed_config()
    maintenance = config['watchdog'] if config else {}
    if units is None and config:
        units = maintenance['units']
    try:
        lifecycle_state = state['jobs'].get('task_lifecycle', {})
        previous_check = float(lifecycle_state.get('scan_through', lifecycle_state.get('last_success_at', alerts.get('last_check_unix', check_time - 600))))
    except (TypeError, ValueError):
        previous_check = check_time - 600
    context = _WatchdogContext(task_db, cache, _unit_names(units), previous_check, unit_checker, maintenance)
    pending_tasks = set(str(value) for value in alerts.get('pending_tasks', []))
    unit_failures: list[tuple[str, str]] = []
    job_errors: list[str] = []
    unit_probe_ran = False

    selected_jobs = tuple(jobs if jobs is not None else default_jobs(maintenance if config else None))
    for job in selected_jobs:
        if job.name == 'git_auto_update':
            continue  # Updating code must be the very last operation in this process.
        job_state = state['jobs'].setdefault(job.name, {})
        if not _job_due(job_state, job.interval_seconds, check_time):
            continue
        started_at = time.time() if now is None else check_time
        try:
            result = job.runner(context)
        except Exception as exc:
            _record_job(
                state, job, started_at=started_at, finished_at=time.time() if now is None else check_time,
                result=None, error=exc,
            )
            job_errors.append(f'{job.name}: {type(exc).__name__}: {exc}')
            continue
        _record_job(
            state, job, started_at=started_at, finished_at=time.time() if now is None else check_time,
            result=result,
        )
        if job.name == 'task_lifecycle':
            state['jobs'][job.name]['scan_through'] = started_at
        pending_tasks.update(result.pending_tasks)
        unit_failures.extend(result.unit_failures)
        unit_probe_ran = unit_probe_ran or job.name == 'systemd_unit_probe'

    unit_alerted = {str(key): bool(value) for key, value in alerts.get('unit_alerted', {}).items()}
    if unit_probe_ran:
        failed_units = {unit for unit, _ in unit_failures}
        for unit in context.units:
            if unit not in failed_units:
                unit_alerted[unit] = False
    alerts.update({
        'pending_tasks': sorted(pending_tasks),
        'unit_alerted': unit_alerted,
        'last_check': datetime.now().astimezone().isoformat(timespec='seconds'),
        'last_check_unix': check_time,
    })
    state.update({'last_run_at': check_time, 'last_job_errors': job_errors})
    _save_state(state_path, state)
    delivered = _deliver_alerts(
        task_db=task_db, state=state, unit_failures=unit_failures,
        state_path=state_path, email_sender=email_sender,
    )
    from src.api.task_monitor.scheduling.runtime import deliver_timeout_events
    if email_sender is None:
        from src.proj.util.web.emailer import Email
        email_sender = Email.send
    timeout_delivered = deliver_timeout_events(email_sender)
    from src.api.task_monitor.scheduling.git_update import deliver_update_emails
    git_mail_delivered = deliver_update_emails(email_sender)
    for job in selected_jobs:
        if job.name != 'git_auto_update' or not _job_due(state['jobs'].get(job.name, {}), job.interval_seconds, check_time):
            continue
        started_at = time.time() if now is None else check_time
        try:
            result = job.runner(context)
            error = None
        except Exception as exc:
            result, error = None, exc
        _record_job(state, job, started_at=started_at, finished_at=time.time() if now is None else check_time,
                    result=result, error=error)
        # Failures of this optional network job obey its configured interval too.
        state['jobs'][job.name]['last_attempt_completed_at'] = time.time() if now is None else check_time
        if error is not None or (result and result.stats.get('status') == 'error'):
            reason = str(error) if error is not None else str(result.stats.get('reason') if result is not None else 'unknown failure')
            state['jobs'][job.name]['last_error'] = reason
            job_errors.append(f'{job.name}: {reason}')
        state['last_job_errors'] = job_errors
        _save_state(state_path, state)
        if result is not None and result.stats.get('notification_id'):
            sent = deliver_update_emails(email_sender, notice_id=str(result.stats['notification_id']))
            git_mail_delivered = git_mail_delivered and sent
    return delivered and timeout_delivered and git_mail_delivered and not job_errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Learndl watchdog and due short-maintenance jobs')
    parser.add_argument('--unit', action='append', default=None, help='Expected active systemd unit (repeatable)')
    parser.add_argument('--state-path', type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        with ScriptLock('learndl_watchdog', timeout=1, wait_time=2):
            return 0 if run_watchdog(units=args.unit, state_path=args.state_path) else 1
    except (BlockingIOError, TimeoutError):
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
