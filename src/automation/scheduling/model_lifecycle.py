"""Schedule explicit model promotions through the supervised idle worker."""
from __future__ import annotations

from .recovery import record, recovery_plan

TASK = 'model_promotion'


def candidate(store, options):
    import portalocker
    from src.proj import Const
    from src.res.model.util import lifecycle_promotion
    from .idle_worklist import OWNER, ACTIVE, version

    declarations = Const.Model.strategies.get('lifecycle', {})
    if not isinstance(declarations, dict):
        message = 'strategy/model.yaml lifecycle must be a mapping'
        store.event({'id': 'promotion-config-' + version({'error': message}), 'owner': OWNER,
                     'task_kind': TASK}, 'configuration_error', message)
        return None
    owned = store.owned(OWNER)
    for name, options_for_model in declarations.items():
        if isinstance(options_for_model, dict) and set(options_for_model) <= {'start', 'end'}:
            continue  # Production date bounds are handled by the nightly task.
        try:
            plan = lifecycle_promotion.plan(name, options_for_model)
        except portalocker.LockException:
            continue
        except Exception as exc:
            message = f'Lifecycle promotion {name}: {type(exc).__name__}: {exc}'
            key = version({'model': name, 'declaration': options_for_model, 'error': message})
            store.event({'id': f'promotion-config-{key}', 'owner': OWNER, 'task_kind': TASK},
                        'configuration_error', message)
            continue
        if plan is None:
            continue
        task = f'promotion:{name}'
        previous = next((run for run in reversed(owned) if run.get('task_kind') == TASK
                         and run['task'] == task and run['version'] == plan['version']
                         and run['phase'] != 'deferred'), None)
        recovery = {}
        if previous:
            if previous['phase'] in ACTIVE or previous['phase'] == 'complete':
                continue
            decision = recovery_plan(store, previous, options)
            if not decision['retry']:
                continue
            plan = previous['promotion_request']
            recovery = {key: decision[key] for key in ('parent_run_id', 'retry_index', 'retry_not_before')}
        return {'task': task, 'task_kind': TASK, 'version': plan['version'],
                'promotion_request': plan, 'script_email_enabled': False, **recovery}
    return None


def execute(request, store):
    from src.res.model.model_module.application import ModelTrainer
    from src.res.model.util import lifecycle_promotion

    plan = request['promotion_request']
    try:
        result = lifecycle_promotion.execute(plan, ModelTrainer.GO)
    except BaseException as exc:
        outcome = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'error'
        record(store, request['id'], script_outcome=outcome, error=f'{type(exc).__name__}: {exc}')
        raise
    record(store, request['id'], script_outcome='success', promotion_candidate=result, promotion_stage=plan['stage'])
    return 0
