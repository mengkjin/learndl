"""Adapt model lifecycle repair plans to the existing idle training lane."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.res.model.util.lifecycle import ModelLifecycle

__all__ = ['candidate', 'execute']

def candidate(store, options: Mapping) -> dict[str, Any] | None:
    """Return a scheduler-neutral repair request with bounded recovery metadata."""
    if not options.get('enabled'):
        return None
    from src.automation.scheduling.recovery import recovery_plan
    for request in ModelLifecycle.inspect_registered():
        task = f'lifecycle:{request["name"]}'
        previous = next((r for r in reversed(store.owned('idle_worklist'))
                         if r.get('task') == task and r.get('version') == request['version']
                         and r.get('phase') != 'deferred'), None)
        recovery = {}
        if previous:
            if previous['phase'] in ('complete', 'queued', 'starting', 'running'):
                continue
            if previous.get('suppress_auto_recovery') and not previous.get('retry_allowed'):
                continue
            recovery = recovery_plan(store, previous, dict(options))
            if not recovery['retry']:
                continue
        return {'task_kind': ModelLifecycle.TASK, 'task': task, 'version': request['version'],
                'lifecycle_request': request,
                **{k: recovery[k] for k in ('parent_run_id', 'retry_index', 'retry_not_before') if k in recovery}}
    return None


def execute(request: dict, store) -> int:
    from .recovery import record
    from src.res.model.model_module.application.trainer import ModelTrainer
    try:
        plan = request['lifecycle_request']
        ModelLifecycle(Path(plan['base'])).repair(plan, ModelTrainer.GO)
    except BaseException as exc:
        record(store, request['id'], script_outcome='error', error=f'{type(exc).__name__}: {exc}')
        raise
    record(store, request['id'], script_outcome='success')
    return 0
