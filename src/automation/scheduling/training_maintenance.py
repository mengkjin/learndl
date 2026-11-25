"""Maintenance requests and durable continuations for checkpoint-aware training."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import sys
import time
import uuid

from src.shared.file_ops import atomic_json
from .config import installed_config, runtime_dir

__all__ = [
    'TASK', 'ENTRIES', 'TrainingPaused', 'available', 'publish_request', 'maintenance_request',
    'worklist_entry', 'continuation', 'checkpointed_script', 'checkpoint_group', 'checkpoint',
    'enqueue', 'candidate', 'cancel', 'record_pause', 'configure_resume', 'validate_checkpoint',
    'validate_active_checkpoint', 'previous_training_run', 'validate_remaining_nodes', 'execute',
]

TASK = 'maintenance_resume'
ENTRIES = {'model': '1_train_model.py', 'schedule': '2_schedule_model.py'}
_entry: ContextVar[str | None] = ContextVar('maintenance_training_entry', default=None)
_worklist: ContextVar[dict | None] = ContextVar('maintenance_training_worklist', default=None)
_resume: ContextVar[dict | None] = ContextVar('maintenance_training_resume', default=None)
_promotion: ContextVar[dict | None] = ContextVar('maintenance_promotion', default=None)


class TrainingPaused(BaseException):
    """Unwind the fit pipeline and ScriptTool without completing the training."""

    def __init__(self, ticket):
        self.ticket = ticket
        super().__init__(f'Training paused for {ticket["maintenance"]["kind"]}')


def available():
    config = installed_config() or {}
    jobs = config.get('watchdog', {}).get('jobs', {})
    return sys.platform == 'linux' and config.get('scheduler_version', 1) >= 3 and (
        config.get('training_maintenance') or any(jobs.get(name, {}).get('enabled')
                                                  for name in ('idle_worklist', 'rl_experiment')))


def publish_request(kind, identity=None, *, directory=None, interval=60):
    """Called by the maintenance worker after its normal validation."""
    path = (directory or runtime_dir()) / f'training-{kind}-request.json'
    if identity is None:
        path.unlink(missing_ok=True)
    else:
        atomic_json(path, {'kind': kind, 'id': identity, 'expires_at': time.time() + max(180, interval * 3)})


def maintenance_request(group):
    from . import install_request
    from .runtime import boot_id

    row = install_request.current()
    impact = row.get('admission_plan', {}).get('impact', {})
    groups = row.get('groups')
    if (row.get('phase') == 'waiting' and not row.get('force') and row.get('boot_id') == boot_id()
            and row.get('expires_at', 0) > time.time()
            and impact.get('mode', 'exclusive') == 'exclusive'
            and (groups is None or group in groups)):
        return {'kind': 'installation', 'id': row['id']}
    for kind in ('patch', 'git'):
        try:
            request = json.loads((runtime_dir() / f'training-{kind}-request.json').read_text())
            if request['expires_at'] > time.time():
                return request
        except FileNotFoundError:
            continue
    from src.proj.util.script.fit_lock import FitLock
    if FitLock.production_requested():
        return {'kind': 'production_model_update', 'id': 'production-fit'}
    return None


@contextmanager
def worklist_entry(details):
    token = _worklist.set(details)
    try:
        yield
    finally:
        _worklist.reset(token)


@contextmanager
def continuation(ticket):
    token = _resume.set(ticket)
    try:
        yield
    finally:
        _resume.reset(token)


@contextmanager
def promotion_entry(plan):
    token = _promotion.set(plan)
    try:
        yield
    finally:
        _promotion.reset(token)


def checkpointed_script(kind):
    """Register only standard training entry points. A batch owns its own exit."""
    def decorate[**P, R](func: Callable[P, R]) -> Callable[P, R | TrainingPaused]:
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R | TrainingPaused:
            token = _entry.set(kind)
            try:
                return func(*args, **kwargs)
            except TrainingPaused as pause:
                if _worklist.get() is not None or _resume.get() is not None:
                    raise
                enqueue(pause)
                return pause
            finally:
                _entry.reset(token)
        return wrapped
    return decorate


def checkpoint_group():
    """Only explicit CLI entries and the managed worklist can hand off a fit."""
    from . import training_scheduler

    admission = training_scheduler.current()
    if _promotion.get() is not None and admission and admission['mode'] == 'unified':
        return 'research'
    if _entry.get() is None or not available() or os.environ.get('LEARNDL_AUTOFIX_ID'):
        return None
    managed = os.environ.get('LEARNDL_MANAGED_RUN')
    if managed and _worklist.get() is None and _resume.get() is None:
        return None
    return 'research' if managed else 'manual'


def checkpoint(trainer):
    from . import training_scheduler

    group = checkpoint_group()
    if group is None:
        return
    kind, worklist = _entry.get(), _worklist.get()
    managed = os.environ.get('LEARNDL_MANAGED_RUN')
    from src.proj import Save
    from src.automation.artifact_exchange.research import CURRENT_TASK
    from src.res.model.util.lifecycle import ModelLifecycle, _checksum

    config = trainer.config
    if config.module_type not in ('nn', 'boost') or config.is_null_model:
        return
    priority_request = training_scheduler.pause_request()
    request = maintenance_request(group) or priority_request
    if request is None:
        return
    Save.async_wait_all(raise_first_error=True, caller_name='maintenance checkpoint')
    date, number = int(trainer.model_date), int(trainer.model_num)
    manager = ModelLifecycle(config.base_path.base)
    files = {}
    required = manager.required_weights(config)
    completed = []
    for fitted_date in trainer.data.model_date_list:
        for fitted_number in config.model_num_list:
            if (fitted_date, fitted_number) > (date, number):
                break
            completed.append([int(fitted_date), int(fitted_number)])
            for submodel in config.submodels:
                weights = manager.weights(fitted_number, fitted_date, submodel)
                if not required.issubset(weights):
                    raise ValueError(f'Incomplete maintenance checkpoint: {fitted_date}/{fitted_number}/{submodel}')
                files.update({f'archive/{fitted_number}/{fitted_date}/{submodel}/{name}': digest
                              for name, digest in weights.items()})
    run = trainer.training_run
    trainer.record.purge_retrained_model_preds()
    ticket = {
        'id': run.run_id, 'entry': kind, 'managed_parent': managed, 'worklist': worklist,
        'maintenance': request, 'model_path': str(manager.base.absolute()), 'weights': files,
        'node': [date, number], 'completed': completed, 'created_at': time.time(),
        'retrained': [[int(d), int(n)] for d, n in trainer.record.retrained_models],
        'config_files': {str(path.relative_to(manager.base)): _checksum(path)
                         for path in config.base_path.conf().glob('*.yaml')
                         if _promotion.get() is None or path.name != 'lifecycle.yaml'},
        'start': config.beg_date, 'end': config.end_date,
        'stage': 0 if 'test' in config.queue_of_stages else 1,
        'short_test': config.short_test,
        'email': CURRENT_TASK.get().email if CURRENT_TASK.get() is not None else True,
        'schedule_name': worklist['reference'] if worklist else (
            config.model_config.schedule_name or (_resume.get() or {}).get('schedule_name')),
    }
    admission = training_scheduler.current()
    if admission and (admission['mode'] == 'unified' or (_resume.get() or {}).get('training_priority')):
        ticket['training_priority'] = admission['kind']
        ticket['training_requested_at'] = admission['requested_at']
        ticket['training_resource'] = admission['resource']
    if promotion := _promotion.get():
        ticket.update(entry='promotion', promotion_request=promotion, email=False)
    _save_ticket(ticket, ready=False)
    trainer.logger.note(f'Paused after {date}/{number} for {request["kind"]}. Resume ID: {run.run_id}')
    raise TrainingPaused(ticket)


def enqueue(pause):
    """Publish only after the trainer and the script recorder have unwound."""
    _save_ticket(pause.ticket, ready=True)


def _save_ticket(ticket, *, ready):
    atomic_json(runtime_dir() / 'maintenance-resumes' / f'{ticket["id"]}.json', dict(ticket, ready=ready))


def _parent_released(store, ticket):
    if parent_id := ticket.get('managed_parent'):
        from .recovery import RECOVERABLE_EXITS, exit_reason

        parent = store.get(parent_id)
        if parent.get('phase') in {'starting', 'running', 'terminating', 'finalizing'}:
            return False
        if not parent or parent.get('suppress_auto_recovery') or 'term_at' in parent:
            return False
        outcome = exit_reason(store, parent)
        return (parent.get('returncode') == 75 or outcome == 'maintenance_pause'
                or bool(ticket.get('training_priority') and outcome in RECOVERABLE_EXITS))
    # A direct CLI call can return to a live menu. Only the synchronous script
    # scope needs to have exited, which is what publication above acknowledges.
    return True


def candidates(store):
    from .idle_worklist import OWNER, settings, version
    from .recovery import recovery_plan

    folder = runtime_dir() / 'maintenance-resumes'
    attempts = None
    for path in sorted(folder.glob('*.json'), key=lambda path: (path.stat().st_mtime_ns, path.name)):
        ticket = json.loads(path.read_text())
        if not ticket['ready'] and not (ticket.get('training_priority') and ticket.get('managed_parent')):
            continue
        if not _parent_released(store, ticket):
            continue
        identity = ticket['id']
        existing = store.get(identity)
        if ticket.get('training_priority'):
            if attempts is None:
                attempts = store.owned(OWNER)
            latest = next((run for run in reversed(attempts) if run.get('resume_ticket_id') == identity), None)
            existing = latest or existing
        retry = {}
        if existing:
            if existing['phase'] == 'deferred' and not existing.get('started_at'):
                identity = existing['id']
                retry = {key: existing[key] for key in ('parent_run_id', 'retry_index', 'retry_not_before') if key in existing}
            elif ticket.get('training_priority') and existing['phase'] == 'error':
                decision = recovery_plan(store, existing, settings())
                if not decision['retry']:
                    continue
                retry = {key: decision[key] for key in ('parent_run_id', 'retry_index', 'retry_not_before')}
                identity = uuid.uuid5(uuid.NAMESPACE_URL, f'training-resume:{ticket["id"]}:{existing["id"]}').hex
            else:
                continue
        worklist = ticket['worklist']
        parent = store.get(ticket['managed_parent']) if ticket['managed_parent'] else {}
        promotion = ticket.get('promotion_request')
        automatic = worklist and worklist.get('automatic_source', bool(ticket.get('managed_parent')))
        yield {'id': identity, 'task_kind': TASK,
               'task': f'promotion:{promotion["name"]}' if promotion else (
                   worklist['name'] if worklist else ticket['schedule_name'] or Path(ticket['model_path']).name),
               'schedule_reference': ticket['schedule_name'],
               'version': version(worklist['revisions']) if worklist else ticket['id'],
               'revisions': worklist['revisions'] if worklist else {},
               'training_priority': ticket.get('training_priority', 'promotion' if promotion else 'idle' if automatic else 'manual'),
               'training_requested_at': ticket.get('training_requested_at', ticket.get('created_at', path.stat().st_mtime)),
               'training_resource': ticket.get('training_resource', 'all'),
               'retry_index': parent.get('retry_index', 0), 'no_email': not ticket['email'] and not promotion,
               **({'resume_ticket_id': ticket['id']} if ticket.get('training_priority') else {}),
               **retry,
               **({'script_email_enabled': False} if promotion else {})}


def candidate(store, *, unified=False):
    waiting = candidates(store)
    if not unified:
        return next(waiting, None)
    from .training_scheduler import PRIORITY
    return min(waiting, key=lambda row: (PRIORITY[row['training_priority']], row['training_requested_at'], row['id']),
               default=None)


def cancel(store, ticket_id):
    from .idle_worklist import OWNER, queue_lock

    if len(ticket_id) != 32 or any(c not in '0123456789abcdef' for c in ticket_id):
        raise ValueError('Expected a maintenance resume ID')
    with queue_lock():
        path = runtime_dir() / 'maintenance-resumes' / f'{ticket_id}.json'
        ticket = json.loads(path.read_text())
        run = store.get(ticket_id)
        if not ticket['ready'] or (run and (run.get('started_at') or run['phase'] == 'cancelled')):
            raise ValueError('Maintenance continuation is no longer waiting')
        store.put(dict(run, id=ticket_id, owner=OWNER, task_kind=TASK, phase='cancelled',
                       task=(ticket['worklist'] or {}).get('name') or Path(ticket['model_path']).name,
                       suppress_auto_recovery=True, finished_at=time.time()))


def record_pause(store, run_id):
    from .recovery import record
    record(store, run_id, script_outcome='maintenance_paused')


def configure_resume(config_kwargs, model_kwargs):
    ticket = _resume.get()
    if ticket is None:
        return
    validate_checkpoint(ticket, 'config_files')
    # Archive YAMLs own the resumed configuration. Worklist versions were
    # checked separately; no current schedule can override the saved options.
    config_kwargs.update(base_path=Path(ticket['model_path']), module=None, schedule_name=None)
    model_kwargs.update(resume=1, selection=0, stage=ticket['stage'], start=ticket['start'],
                        end=ticket['end'], short_test=ticket['short_test'])
    model_kwargs.pop('lifecycle_options', None)


def validate_checkpoint(ticket, field='weights'):
    from src.res.model.util.lifecycle import _checksum

    base = Path(ticket['model_path'])
    if not base.is_dir():
        raise FileNotFoundError(f'Maintenance resume directory is missing: {base}')
    for relative, digest in ticket[field].items():
        path = base / relative
        if not path.is_file() or _checksum(path) != digest:
            raise ValueError(f'Maintenance resume checkpoint is missing or changed: {path}')


def validate_active_checkpoint() -> list[tuple[int, int]]:
    if ticket := _resume.get():
        validate_checkpoint(ticket)
        # Older tickets only recorded the completed prefix, including reused weights.
        return [(int(d), int(n)) for d, n in ticket.get('retrained', ticket['completed'])]
    return []


def previous_training_run():
    if ticket := _resume.get():
        return ticket['id']
    return None


def validate_remaining_nodes(nodes, pending):
    if ticket := _resume.get():
        completed = {tuple(node) for node in ticket['completed']}
        if completed.difference(nodes) or completed.intersection(pending):
            raise ValueError('Saved training nodes no longer match the model configuration or data')


def execute(request):
    """Run under the normal idle training admission and supervision."""
    from src.automation.paths import ROOT
    from src.proj.util.cli.script_session import as_script_main
    from src.res.model.workflow.execution import ScheduleWorklist
    from src.shared.dynamic_import import dynamic_modules

    ticket_id = request.get('resume_ticket_id', request['id'])
    ticket = json.loads((runtime_dir() / 'maintenance-resumes' / f'{ticket_id}.json').read_text())
    if ticket['entry'] == 'promotion':
        from . import model_lifecycle
        from .runtime import RunStore
        from src.res.model.util.lifecycle_promotion import still_requested
        from src.res.model.workflow.worklist_state import AutomaticDeferred
        if not still_requested(ticket['promotion_request']):
            raise AutomaticDeferred('Paused lifecycle declaration or target was superseded')
        validate_checkpoint(ticket, 'config_files')
        validate_checkpoint(ticket)
        with continuation(ticket):
            return model_lifecycle.execute(
                request | {'promotion_request': ticket['promotion_request']}, RunStore(),
                expected_candidate=Path(ticket['model_path']))
    if ticket['worklist']:
        sys.argv = [str(ScheduleWorklist.SCHEDULE_SCRIPT)]
        return ScheduleWorklist(automatic=True, maintenance_ticket=ticket).run()
    with continuation(ticket):
        script = ROOT / 'scripts' / '4_train' / ENTRIES[ticket['entry']]
        sys.argv = [str(script)]
        with as_script_main(script):
            main = next(iter(dynamic_modules(script))).main
            options = {'email': ticket['email'], 'short_test': ticket['short_test']}
            if ticket['entry'] == 'schedule':
                options.update(schedule_name=ticket['schedule_name'], resume=True,
                               base_path=ticket['model_path'])
            try:
                result = main(**options)
            except TrainingPaused as pause:
                enqueue(pause)
                return pause
        if not (result.success and result.execution_success):
            raise RuntimeError(f'Maintenance continuation failed: {ticket["model_path"]}')
        return None
