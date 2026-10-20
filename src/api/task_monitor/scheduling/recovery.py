"""Bounded abnormal-exit recovery with optional, identity-bound OOM diagnostics."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import psutil

from .runtime import RunStore, boot_id

RECOVERABLE_EXITS = {'oom', 'unknown_exit', 'native_crash', 'reboot'}


def evidence(store: RunStore, run_id: str) -> dict:
    with store.connect() as connection:
        row = connection.execute('SELECT data FROM run_evidence WHERE run_id=?', (run_id,)).fetchone()
    return json.loads(row[0]) if row else {}


def record(store: RunStore, run_id: str, **values) -> None:
    # Separate from runs so watchdog/launcher snapshots cannot erase child results.
    with store.connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        row = connection.execute('SELECT data FROM run_evidence WHERE run_id=?', (run_id,)).fetchone()
        data = json.loads(row[0]) if row else {}
        data.update(values)
        connection.execute('INSERT INTO run_evidence VALUES (?,?) ON CONFLICT(run_id) DO UPDATE SET data=excluded.data',
                           (run_id, json.dumps(data)))


def stop_post(store: RunStore | None = None) -> None:
    """ExecStopPost receives systemd's result for this exact service invocation."""
    store = store or RunStore()
    invocation = os.environ.get('INVOCATION_ID')
    if not invocation:
        return
    for run in store.owned('idle_worklist'):
        if run.get('invocation_id') != invocation or run.get('boot_id') != boot_id():
            continue
        result = os.environ.get('SERVICE_RESULT', 'unknown')
        values = dict(service_result=result, service_exit_code=os.environ.get('EXIT_CODE'),
                      service_exit_status=os.environ.get('EXIT_STATUS'), stop_observed_at=time.time())
        record(store, run['id'], **values)


def child_progress(run_id: str, training: dict, store: RunStore | None = None) -> None:
    store = store or RunStore()
    run = store.get(run_id)
    if run.get('owner') != 'idle_worklist' or run.get('pid') != os.getpid():
        return
    values = {'training_status': training['status'], 'training_run_id': training['run_id']}
    if training.get('model_path'):
        values['model_path'] = training['model_path']
    record(store, run_id, **values)


def script_result(status: str) -> None:
    """Persist normal script outcomes before SMTP, including setup exceptions."""
    run_id = os.environ.get('LEARNDL_MANAGED_RUN')
    if not run_id or status not in {'Success', 'Error'}:
        return
    store = RunStore()
    run = store.get(run_id)
    if run.get('owner') == 'idle_worklist' and run.get('pid') == os.getpid():
        record(store, run_id, script_outcome='success' if status == 'Success' else 'error')


def exit_reason(store: RunStore, run: dict) -> str:
    data = evidence(store, run['id'])
    if 'term_at' in run:
        return 'watchdog_timeout'
    if data.get('script_outcome') == 'error' or data.get('training_status') == 'failed':
        return 'script_error'
    if data.get('script_outcome') == 'success' or data.get('training_status') == 'success' or run.get('returncode') == 0:
        return 'success'
    if data.get('script_outcome') == 'interrupted' or data.get('training_status') == 'interrupted':
        return 'interrupted'
    code = run.get('returncode')
    # Explicit INT/TERM always wins, including after reboot or an earlier OOM.
    if code in (-2, -15, 130, 143) or (
            data.get('service_exit_code') == 'killed' and
            data.get('service_exit_status') in ('TERM', 'INT', 'SIGTERM', 'SIGINT', '15', '2')) or (
            data.get('service_exit_code') == 'exited' and data.get('service_exit_status') in ('130', '143')):
        return 'interrupted'
    if code is not None and code > 0:
        return 'script_error'
    if run.get('boot_id') and run['boot_id'] != boot_id():
        return 'reboot'
    if code is not None and code < 0 and code != -9:
        return 'native_crash'
    if data.get('service_result') == 'oom-kill':
        return 'oom'
    return 'unknown_exit'


def recovery_plan(store: RunStore, run: dict, options: dict, *, now: float | None = None) -> dict:
    if run.get('task_kind') == 'rl_experiment':
        return dict(retry=False, recovery_status='RL attempts require a committed configuration/rerun_mark change')
    now = time.time() if now is None else now
    reason = exit_reason(store, run)
    count = int(run.get('retry_index', 0))
    result = {'exit_reason': reason, 'retry': False, 'attempted_retries': count,
              'max_retries': options.get('recovery_max_retries', 3)}
    if reason not in RECOVERABLE_EXITS:
        return dict(result, recovery_status='exit is not eligible for automatic recovery')
    if not options.get('recovery_enabled', True):
        return dict(result, recovery_status='abnormal-exit recovery disabled')
    if count >= options.get('recovery_max_retries', 3):
        return dict(result, recovery_status='retry limit reached')
    path = evidence(store, run['id']).get('model_path')
    if not path or not Path(path).is_dir():
        return dict(result, recovery_status='cannot resume: recorded training directory missing')
    due = run.get('finished_at', now) + options.get('recovery_delay_seconds', 900)
    result.update(parent_run_id=run['id'], retry_index=count + 1, resume_model_path=path, retry_not_before=due)
    if now < due:
        return dict(result, recovery_status='cooldown')
    try:
        memory = psutil.virtual_memory()
        available = memory.available / memory.total * 100
    except (OSError, ValueError, ZeroDivisionError):
        return dict(result, recovery_status='system memory unavailable')
    if available < options.get('recovery_min_available_memory_percent', 20):
        return dict(result, recovery_status='system memory busy')
    return dict(result, retry=True, recovery_status='ready')


def resume_path(store: RunStore, request: dict, task: str, version: str) -> str:
    """Recheck the immutable parent and exact path inside the schedule lock."""
    parent = store.get(request.get('parent_run_id', ''))
    path = request.get('resume_model_path')
    if (request.get('pid') != os.getpid() or request.get('task') != task or request.get('version') != version
            or parent.get('task') != task or parent.get('version') != version
            or exit_reason(store, parent) not in RECOVERABLE_EXITS
            or path != evidence(store, parent['id']).get('model_path')):
        raise RuntimeError('Abnormal-exit recovery identity/configuration changed')
    if not path or not Path(path).is_dir():
        raise FileNotFoundError('Cannot resume: recorded training directory missing')
    return path


if __name__ == '__main__':
    stop_post()
