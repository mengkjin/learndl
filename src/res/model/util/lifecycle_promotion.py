"""One-shot lifecycle upgrades in an isolated, indexed model directory."""
from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from .core import ModelPath
from .lifecycle import ModelLifecycle, _digest, _read, _target_digest, _write


def registration(name, options):
    if not isinstance(options, Mapping) or set(options) - {'stage', 'start', 'end', 'rerun_mark'}:
        raise ValueError(f'Invalid lifecycle promotion fields for {name}')
    if options.get('stage') not in ('validation', 'production'):
        raise ValueError(f'Expected validation or production promotion stage for {name}')
    if not isinstance(options.get('rerun_mark', 0), (str, int)):
        raise ValueError(f'Invalid lifecycle promotion rerun_mark for {name}')
    for key in ('start', 'end'):
        if key in options and (type(options[key]) is not int or not 19000101 <= options[key] <= 99991231):
            raise ValueError(f'Invalid promotion {key} date for {name}')
    if options.get('start', 0) > options.get('end', 99991231):
        raise ValueError(f'Promotion starts after its end date for {name}')
    source = ModelPath(name)
    manager = ModelLifecycle(source.base)
    if source.is_short_test or not source.requires_fit:
        raise ValueError(f'Expected a non-short-test NN or boosting model for {name}')
    if manager.is_registered() or manager.state.get('promotion_version'):
        raise ValueError(f'Promotion requires an exploration source directory: {name}')
    if not source.conf_file('model').is_file():
        raise FileNotFoundError(f'Promotion source configuration is missing: {source.base}')
    return dict(options) | {'name': name, 'base': str(source.base)}


def plan(name, options):
    from src.proj import CALENDAR
    from .config import ModelConfig

    declared = registration(name, options)
    manager = ModelLifecycle(Path(declared['base']))
    with manager.lock():
        # Upgrade legacy directories from their saved configuration on first use.
        if not manager.file.exists():
            saved = ModelConfig(ModelPath(manager.base), stage=2, resume=1, selection=0,
                                short_test=False, vb_level='never')
            manager.prepare(saved)
        state = manager.state
        identity = _digest((declared, _target_digest(_read(manager.target_file)), state.get('repair_generation', 0)))
        if state.get('upgrade_accepted') == identity:
            return None
        request, config, check_data_cutoff = manager.repair_plan(declared, state)
        if not request['nodes']:
            return None
        if check_data_cutoff and CALENDAR.updated() < manager.fit_sample_cutoff(config, request['nodes'][-1]):
            return None
        # Calendar changes alone must not turn an explicit upgrade into a recurring job.
        return request | {'version': identity}


def _check_request(source, request):
    from src.proj import Const

    name = request['name']
    options = Const.Model.strategies.get('lifecycle', {}).get(name)
    if _digest(registration(name, options)) != request['registration_version']:
        raise ValueError(f'Lifecycle promotion declaration changed for {name}')
    target = _read(source.target_file)
    if (_target_digest(target) != request['target_version']
            or source.state.get('repair_generation', 0) != request['repair_generation']):
        raise ValueError(f'Lifecycle promotion target changed for {name}')
    return target


def _new_candidate(source, request, target):
    path = ModelPath(source.base)
    path.with_new_index(path.find_new_index(folder_not_exist=True))
    registered = ModelLifecycle.production_names()
    while path.full_name in registered or path.base.exists():
        path.with_new_index(path.model_name_index + 1)
    path.base.mkdir(parents=True, exist_ok=False)
    _write(path.conf_file('model'), target['model'])
    _write(path.conf_file('schedule'), {})
    _write(path.conf_file(f'algo.{path.model_module}'), target['algo'])
    if target.get('boost_head'):
        head = target['model']['model.module.nn.boost_head.boost']
        _write(path.conf_file(f'algo.{head}'), target['boost_head'])
    staged = ModelLifecycle(path.base)
    _write(staged.target_file, target)
    staged.save(stage='validation', status='upgrading', legacy=False, budget={'enabled': False},
                target_version=request['target_version'], origin_model=ModelPath(source.base).full_name,
                promotion_stage=request['stage'], promotion_version=request['version'])
    source.save(upgrade_candidate=str(path.base), upgrade_status='upgrading', upgrade_error=None)
    return staged


def _copy_weights(source, staged, request, target):
    model = target['model']
    if source.state.get('force_retrain') or model.get('train.trainer.transfer'):
        return
    expected = _digest(ModelLifecycle.semantic(target | {'dependencies': request['dependencies']}))
    copied = {}
    for key, record in _read(source.manifest_file).items():
        number, date, submodel = key.split(':')
        if int(date) not in request['nodes'] or record.get('definition') != expected:
            continue
        weights = source.weights(int(number), int(date), submodel)
        if not weights or weights != record.get('weights'):
            continue
        relative = Path('archive', number, date, submodel)
        shutil.copytree(source.base / relative, staged.base / relative)
        copied[key] = record | {'files': staged.file_stats(int(number), int(date), submodel)}
    _write(staged.manifest_file, copied)


def execute(request, run):
    source = ModelLifecycle(Path(request['base']))
    with source.lock():
        target = _check_request(source, request)
        state = source.state
        recorded = state.get('upgrade_candidate')
        staged = ModelLifecycle(Path(recorded)) if recorded else None
        if staged is not None and staged.state.get('promotion_version') != request['version']:
            staged = None
        if staged is not None and staged.state.get('status') == 'ready':
            source.save(upgrade_accepted=request['version'], upgrade_status='ready', upgrade_error=None)
            return ModelPath(staged.base).full_name
        fresh = staged is None
        if fresh:
            staged = _new_candidate(source, request, target)
        assert staged is not None
        with staged.lock():
            if staged.is_registered():
                raise ValueError(f'Candidate was registered before acceptance: {staged.base}')
            staged.save(status='upgrading')
            try:
                if fresh:
                    _copy_weights(source, staged, request, target)
                trainer = staged.fit_and_validate(
                    request, run, title=f'Lifecycle upgrade of {request["name"]}', operation='lifecycle_upgrade',
                    lifecycle_promotion=request['version'])
                if not len(trainer.data.test_full_dates):
                    raise ValueError('Lifecycle promotion requires out-of-sample predictions')
                if staged.dependencies(trainer.config, production=True) != request['dependencies']:
                    raise ValueError('Upstream dependencies changed during lifecycle promotion')
                _check_request(source, request)
                evaluation = {'request': dict(request), 'artifacts': _digest(_read(staged.manifest_file)), 'status': 'passed'}
                _write(staged.base / 'snapshot/evaluation_manifest.json', evaluation)
                staged.save(stage=request['stage'], status='ready', accepted_request=request['version'])
                source.save(upgrade_accepted=request['version'], upgrade_status='ready', upgrade_error=None)
            except BaseException as exc:
                staged.save(status='upgrade_failed')
                source.save(upgrade_status='failed', upgrade_error=f'{type(exc).__name__}: {exc}')
                raise
    return ModelPath(staged.base).full_name
