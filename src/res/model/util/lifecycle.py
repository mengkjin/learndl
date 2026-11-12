"""Model lifecycle policy, portable provenance and isolated production repair plans.

Training and automation execute plans; this module owns their policy and never imports
an application trainer or a scheduler. Production aliases point only to approved runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterator, Mapping, Sequence

import portalocker
import yaml

__all__ = ['ModelLifecycle']


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Expected mapping: {path}')
    return value


def _write(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w') as stream:
            if path.suffix == '.yaml':
                yaml.safe_dump(dict(value), stream, allow_unicode=True, sort_keys=False)
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _checksum(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ModelLifecycle:
    """One model's policy. Runtime dictionaries contain serialized config values."""

    STAGES = ('exploration', 'validation', 'production')
    TASK = 'model_lifecycle'

    def __init__(self, base: Path):
        self.base = Path(base)
        self.file = self.base / 'configs/lifecycle.yaml'
        self.target_file = self.base / 'configs/production.yaml'
        self.manifest_file = self.base / 'snapshot/model_manifest.json'

    @property
    def state(self) -> dict[str, Any]:
        return _read(self.file)

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.base.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.base / '.lifecycle.lock'), timeout=0):
            yield

    def save(self, **changes) -> None:
        state = self.state | changes
        _write(self.file, state)
        path = self.base / 'logs/lifecycle.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        with path.open('a') as stream:
            stream.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), **changes}, default=str) + '\n')

    @staticmethod
    def snapshot(config) -> dict[str, Any]:
        return {'model': dict(config.model_config.Param),
                'algo': dict(config.algo_config.Param),
                'boost_head': dict(config.boost_head_config.Param) if config.boost_head_config else None}

    @staticmethod
    def semantic(snapshot: Mapping) -> dict[str, Any]:
        ignored = {'model.name', 'model.beg_date', 'model.end_date', 'model.interval',
                   'model.interval.nn', 'model.interval.boost'}
        model = {k: v for k, v in snapshot['model'].items()
                 if k not in ignored and not k.startswith(('env.verbosity', 'model.resuming_options',
                                                           'train.torch_compile'))}
        module = str(model.get('model.module', ''))
        if module in ('nn', 'boost'):
            module = str(model.get(f'model.module.{module}', module))
        model['model.module'] = module.split('@')[-1]
        return {'model': model, 'algo': snapshot['algo'], 'boost_head': snapshot.get('boost_head'),
                'dependencies': snapshot.get('dependencies', {})}

    @classmethod
    def validate_entry(cls, config) -> None:
        state = cls(config.base_path.base).state
        if config.lifecycle_repair:
            if not state.get('origin') or state.get('stage') != 'validation':
                raise ValueError('Repair resume requires an isolated validation candidate')
        elif 'fit' in config.queue_of_stages and state.get('stage') == 'production':
            raise ValueError('Production weights are immutable; use lifecycle repair')

    @classmethod
    def dump_training_config(cls, config, kwargs: dict) -> None:
        manager = cls(config.base_path.base)
        if manager.state:
            path = config.base_path.conf_file('model')
            if not path.exists() or kwargs.get('overwrite'):
                _write(path, _read(manager.target_file)['model'])
        else:
            config.model_config.Param.dump_yaml(config.base_path.conf_file('model'), **kwargs)

    @classmethod
    def training_nodes(cls, config, available_dates) -> list[int]:
        from src.proj import CALENDAR
        if not cls(config.base_path.base).state:
            return [int(d) for d in available_dates[::config.interval]]
        anchor = config.model_config.beg_date
        grid = CALENDAR.range(anchor, int(max(available_dates)), 'td')[::config.interval]
        available = set(int(d) for d in available_dates)
        return [d for d in cls.selected_nodes(config, grid.tolist()) if d in available]

    @staticmethod
    def training_options(mode: str | None = None, options: Mapping | None = None) -> dict[str, Any]:
        """Validate exploration controls shared by scripts, worklists and configuration."""
        if options is not None and not isinstance(options, Mapping):
            raise ValueError('lifecycle options must be a mapping')
        result = dict(options or {})
        unknown = set(result) - {'mode', 'step', 'max_epoch', 'window'}
        if unknown:
            raise ValueError(f'Unknown lifecycle training options: {sorted(unknown)}')
        if mode is not None and mode != 'auto':
            if 'mode' in result and result['mode'] != mode:
                raise ValueError('Conflicting training_mode and lifecycle.mode')
            result['mode'] = mode
        if 'mode' in result and result['mode'] not in ('auto', 'quick', 'sparse', 'full'):
            raise ValueError('Training mode must be auto, quick, sparse or full')
        for key in ('step', 'max_epoch', 'window'):
            if key in result and (isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1):
                raise ValueError(f'lifecycle.{key} must be a positive integer')
        return result

    @staticmethod
    def budget_mode(budget: Mapping) -> str:
        return str(budget.get('mode', 'quick' if budget.get('enabled') else 'full'))

    def invalidate_evaluation(self) -> None:
        """Archive derived outputs before a budget change invalidates their routing."""
        import tarfile
        paths = [self.base / 'snapshot' / name for name in ('pred_recorder', 'basic_test', 'detailed_alpha')]
        paths.append(self.base / 'results')
        paths = [path for path in paths if path.exists()]
        if not paths:
            return
        folder = self.base / 'snapshot'
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / f'lifecycle_previous_evaluation.{uuid.uuid4().hex}.tar'
        temporary = archive.with_suffix('.tmp')
        try:
            with tarfile.open(temporary, 'w') as stream:
                for path in paths:
                    stream.add(path, arcname=str(path.relative_to(self.base)))
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
        for path in paths:
            shutil.rmtree(path)

    @classmethod
    def prepare(cls, config) -> None:
        """Called after path selection and before dumping configs or callbacks."""
        if config.module_type not in ('nn', 'boost') or config.is_null_model:
            return
        manager = cls(config.base_path.base)
        state = manager.state
        if (not state and not config.model_config.schedule_name and not config.short_test
                and not config.base_path.conf_file('model').exists()):
            return
        if not state:
            # Existing directories retain their historical budget until explicitly enabled.
            legacy = config.base_path.conf_file('model').exists()
            target = cls.snapshot(config)
            _write(manager.target_file, target)
            manager.save(schema_version=1, stage='exploration', status='legacy' if legacy else 'pending',
                         legacy=legacy, short_test=config.short_test, target_version=_digest(target),
                         budget={'enabled': not legacy and not config.short_test,
                                 'mode': ('quick' if config.module_type == 'nn' else 'sparse') if not legacy and not config.short_test else 'full',
                                 'max_epoch': 20, 'window': 1200, 'step': 2})
            state = manager.state
        if state.get('short_test') and state.get('stage') != 'exploration':
            raise ValueError('Short-test models cannot be promoted')
        requested = dict(config.lifecycle_options)
        if requested.get('mode') == 'auto':
            requested.pop('mode')
        if requested and not config.short_test:
            if state.get('stage') != 'exploration':
                raise ValueError('Exploration training options cannot modify validation/production models')
            budget = dict(state.get('budget', {}))
            mode = requested.pop('mode', cls.budget_mode(budget))
            budget.update(requested, mode=mode, enabled=mode != 'full')
            if budget != state.get('budget'):
                if 'fit' not in config.queue_of_stages:
                    raise ValueError('Changing exploration budget requires a fitting run')
                with manager.lock():
                    manager.invalidate_evaluation()
                    manager.save(budget=budget, status='needs_training')
                state = manager.state
        budget = state.get('budget', {})
        if state.get('stage') == 'exploration' and budget.get('enabled') and not config.short_test:
            for key in ('step', 'max_epoch', 'window'):
                value = budget.get(key, {'step': 2, 'max_epoch': 20, 'window': 1200}[key])
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f'Invalid lifecycle budget {key}: {value}')
            if config.module_type == 'nn' and cls.budget_mode(budget) == 'quick':
                if config.max_epoch > budget.get('max_epoch', 20):
                    config.model_config['train.max_epoch'] = budget.get('max_epoch', 20)
                if config.window > budget.get('window', 1200):
                    config.model_config['model.window'] = budget.get('window', 1200)
        config.value_dict['lifecycle'] = manager
        config.value_dict['lifecycle_budget'] = dict(budget)

    @classmethod
    def selected_nodes(cls, config, dates: Sequence[int]) -> list[int]:
        manager = cls(config.base_path.base)
        state = manager.state
        budget = config.value_dict.get('lifecycle_budget', state.get('budget', {}))
        step = (budget.get('step', 2)
                if state.get('stage') == 'exploration' and budget.get('enabled')
                and not config.short_test else 1)
        return [int(date) for date in dates[::step] if config.beg_date <= date <= config.end_date]

    @classmethod
    def fold_nodes(cls, config, folds: list, dates):
        from .data.factor_sequence_plan import validate_folds
        import numpy as np
        wanted = set(int(d) for d in dates[1:])
        scoped = [dict(fold, test=[int(d) for d in fold['test'] if int(d) in wanted]) for fold in folds]
        scoped = [fold for fold in scoped if fold['test']]
        _, test_dates = validate_folds(scoped, dates)
        selected = set(cls.selected_nodes(config, [int(fold['model_date']) for fold in folds]))
        return np.asarray([fold['model_date'] for fold in scoped if fold['model_date'] in selected], dtype=int), test_dates

    @classmethod
    def compressed_split(cls, config, split: Mapping, available: set[int]) -> dict:
        state = cls(config.base_path.base).state
        result = dict(split)
        budget = config.value_dict.get('lifecycle_budget', state.get('budget', {}))
        if (state.get('stage') == 'exploration' and budget.get('enabled')
                and cls.budget_mode(budget) == 'quick'):
            for group in ('train', 'valid'):
                result[group] = [int(d) for d in split[group] if int(d) in available]
                if not result[group]:
                    raise ValueError(f'Quick window leaves no explicit {group} samples; increase lifecycle window')
        return result

    @classmethod
    def configured(cls, config) -> None:
        manager = cls(config.base_path.base)
        if manager.state:
            effective = cls.snapshot(config)
            effective['dependencies'] = cls.dependencies(config, production=manager.state.get('stage') != 'exploration')
            config.value_dict['lifecycle_effective'] = effective

    @classmethod
    def dependencies(cls, config, *, production: bool) -> dict[str, str]:
        from .model_source import model_source_path
        from .final_source import FinalSource
        references = list(config.input_hidden_types)
        for expression in config.input_final_types:
            source = FinalSource.parse(expression)
            if source.kind == 'model_pred':
                references.append(source.reference)
        signatures = {}
        for reference in references:
            path = model_source_path(reference)
            state = cls(path.base).state
            if production:
                legacy_registered = any(item['base'] == str(path.base) and item['stage'] == 'production'
                                        for item in cls.registrations()) if not state else False
                if (state and state.get('stage') != 'production') or (not state and not legacy_registered):
                    raise ValueError(f'Upstream model is not production-approved: {reference}')
            signatures[reference] = _digest({str(file.relative_to(path.base)): (file.stat().st_size, file.stat().st_mtime_ns)
                                            for file in sorted(path.archive().rglob('*.pt'))})
        return signatures

    def weights(self, number: int, date: int, submodel: str) -> dict[str, str]:
        directory = self.base / 'archive' / str(number) / str(date) / submodel
        files = [path for path in directory.glob('*.pt') if path.name in ('state_dict.pt', 'boost_dict.pt', 'boost_head.pt')]
        return {path.name: _checksum(path) for path in files if path.stat().st_size}

    def file_stats(self, number: int, date: int, submodel: str) -> dict[str, list[int]]:
        directory = self.base / 'archive' / str(number) / str(date) / submodel
        return {file.name: [file.stat().st_size, file.stat().st_mtime_ns]
                for file in directory.glob('*.pt')
                if file.name in ('state_dict.pt', 'boost_dict.pt', 'boost_head.pt')}

    def healthy_record(self, key: str, record: Mapping) -> bool:
        number, date, submodel = key.split(':')
        if record.get('files'):
            return self.file_stats(int(number), int(date), submodel) == record['files']
        return self.weights(int(number), int(date), submodel) == record.get('weights')

    @staticmethod
    def required_weights(config) -> set[str]:
        names = {'state_dict.pt'} if config.module_type == 'nn' else {'boost_dict.pt'}
        if config.boost_head_config:
            names.add('boost_head.pt')
        return names

    def compatible(self, config, number: int, date: int) -> bool:
        manifest = _read(self.manifest_file)
        expected = _digest(self.semantic(config.value_dict.get('lifecycle_effective', self.snapshot(config))))
        for submodel in config.submodels:
            weights = self.weights(number, date, submodel)
            if not self.required_weights(config).issubset(weights):
                return False
            record = manifest.get(f'{number}:{date}:{submodel}')
            if record is None:
                if self.state.get('legacy'):
                    continue
                return False
            if record.get('definition') != expected or record.get('weights') != weights:
                return False
        return True

    @classmethod
    def pending_nodes(cls, trainer, nodes: list[tuple]) -> list[tuple]:
        manager = cls(trainer.base_path.base)
        if not manager.state:
            return [(date, number) for date, number in nodes
                    if not trainer.deposition.exists(number, date)]
        pending = []
        changed_numbers: set[int] = set()
        for date, number in nodes:
            if ((trainer.config.transfer_training and int(number) in changed_numbers)
                    or not manager.compatible(trainer.config, int(number), int(date))):
                pending.append((date, number))
                changed_numbers.add(int(number))
        return pending

    @classmethod
    def record_node(cls, trainer) -> None:
        manager = cls(trainer.base_path.base)
        if not manager.state:
            return
        from src.proj import Save
        Save.async_wait_all(caller_name='ModelLifecycle.record_node')
        config = trainer.config
        manifest = _read(manager.manifest_file)
        for submodel in config.submodels:
            weights = manager.weights(int(trainer.model_num), int(trainer.model_date), submodel)
            if not manager.required_weights(config).issubset(weights):
                raise ValueError(f'Incomplete trained artifact: {trainer.model_date}/{submodel}')
            manifest[f'{trainer.model_num}:{trainer.model_date}:{submodel}'] = {
                'definition': _digest(cls.semantic(config.value_dict['lifecycle_effective'])),
                'effective': config.value_dict['lifecycle_effective'], 'weights': weights,
                'files': manager.file_stats(int(trainer.model_num), int(trainer.model_date), submodel),
                'run_id': trainer.training_run.run_id if trainer.training_run else None,
                'epoch': int(trainer.status.epoch), 'phase': int(trainer.status.phase),
                'attempt': int(trainer.status.attempt), 'previous_date': int(trainer.prev_model_date),
                'data_start': int(trainer.data.datas.date[trainer.data.d0]),
                'data_end': int(trainer.data.datas.date[max(trainer.data.d0, trainer.data.d1 - 1)]),
            }
        _write(manager.manifest_file, manifest)

    @classmethod
    def approved_path(cls, path, *, registered: bool = False):
        """Resolve an approved alias; legacy registrations remain available."""
        manager = cls(path.base)
        state = manager.state
        if not state:
            return path
        approved = state.get('approved_model') or state.get('approved_path')
        if approved:
            result = path.copy()
            result.parse_input(approved if state.get('approved_model') else Path(approved))
            if not result.base.is_dir():
                raise FileNotFoundError(f'Approved model directory is missing: {result.base}')
            return result
        if registered and state.get('legacy'):
            return path
        if registered and state.get('stage') != 'production':
            raise ValueError(f'Model is not approved for production: {path.full_name}')
        return path

    @classmethod
    def registrations(cls) -> list[dict[str, Any]]:
        from src.proj import Const
        from src.res.model.util.core import ModelPath
        strategy = Const.Model.strategies
        entries = dict(strategy.get('lifecycle', {}))
        # Existing prediction registrations are explicit legacy production intent.
        for entry in [*strategy.get('prediction', {}).values(), *strategy.get('hidden_extraction', {}).values()]:
            entries.setdefault(entry['name'], {'stage': 'production'})
        result = []
        for name, entry in entries.items():
            if not isinstance(entry, dict) or entry.get('stage') not in ('validation', 'production'):
                raise ValueError(f'Invalid lifecycle registration: {name}')
            path = ModelPath(name)
            if path.module_type not in ('nn', 'boost'):
                continue  # Non-fitting linear/factor lifecycles retain their existing owners.
            if path.is_short_test:
                raise ValueError(f'Cannot promote short test: {name}')
            result.append({'name': name, 'base': str(path.base), **entry})
        return sorted(result, key=lambda item: item['stage'] != 'production')

    @classmethod
    def inspect_registered(cls) -> list[dict[str, Any]]:
        """Cheap daily reconciliation; no training and no worklist mutations."""
        from src.proj import CALENDAR, Logger
        from src.res.model.util.config import ModelConfig
        from src.res.model.util.core import ModelPath
        requests = []
        for registration in cls.registrations():
            manager = cls(Path(registration['base']))
            try:
                if not manager.base.is_dir():
                    raise FileNotFoundError(f'Model directory missing: {manager.base}')
                with manager.lock():
                    if not manager.file.exists():
                        config = ModelConfig(ModelPath(manager.base), stage=2, resume=1, selection=0, short_test=False)
                        if not config.base_path.conf_file('model').exists():
                            raise FileNotFoundError(f'Model configuration missing: {manager.base}')
                        cls.prepare(config)
                        manager.save(stage='production' if registration['stage'] == 'production' else 'validation',
                                     legacy=True, status='legacy', approved_path=str(manager.base)
                                     if registration['stage'] == 'production' else None)
                    state = manager.state
                    if state.get('legacy') and registration['stage'] == 'production' and state.get('stage') == 'exploration':
                        manager.save(stage='production', approved_path=str(manager.base))
                        state = manager.state
                    if state.get('target_stage') != registration['stage']:
                        manager.save(target_stage=registration['stage'])
                    target = _read(manager.target_file)
                    model = target['model']
                    start = int(registration.get('start', model['model.beg_date']))
                    end = min(int(registration.get('end', model['model.end_date'])), int(CALENDAR.update_to()))
                    target_version = _digest(target)
                    dependency_config = ModelConfig(ModelPath(manager.base), stage=2, resume=1, selection=0,
                                                    override=model, short_test=False, vb_level='never')
                    dependencies = cls.dependencies(dependency_config, production=True)
                    # Calendar cutoff changes only when a new training node becomes due.
                    kind = ModelPath(manager.base).module_type
                    interval = int(model.get('model.interval') or model[f'model.interval.{kind}'])
                    grid = [int(d) for d in CALENDAR.range(int(model['model.beg_date']), end, 'td')[::interval]] if end >= start else []
                    folds = model.get('train.factor_sequence.folds', [])
                    if folds:
                        end = min(end, max(int(d) for fold in folds for d in fold['test']))
                        grid = [int(fold['model_date']) for fold in folds if int(fold['model_date']) <= end]
                    preceding = [d for d in grid if d <= start]
                    training_start = preceding[-1] if preceding else start
                    grid = [d for d in grid if d >= training_start]
                    request = registration | {'registration_version': _digest(registration), 'target_version': target_version, 'dependencies': dependencies, 'start': start,
                                              'repair_generation': int(state.get('repair_generation', 0)),
                                              'end': end, 'training_start': training_start, 'nodes': grid}
                    request['version'] = _digest({k: v for k, v in request.items() if k != 'end'})
                    # Legacy existing artifacts are grandfathered, but new grid nodes still repair.
                    if registration['stage'] == 'validation' and state.get('validated_model'):
                        active = ModelPath(state['validated_model'])
                    else:
                        active = ModelPath(state['approved_model']) if state.get('approved_model') else ModelPath(Path(state.get('approved_path') or str(manager.base)))
                    if state.get('legacy') and not state.get('audit_version'):
                        manager.save(audit_version=request['version'], status='legacy',
                                     audit_note='Historical training evidence incomplete; existing artifacts retained')
                        if grid and all(active.exists(number, date, submodel)
                                    for date in grid for number in dependency_config.model_num_list
                                    for submodel in dependency_config.submodels):
                            manager.save(accepted_request=request['version'])
                            continue
                    if state.get('accepted_request') == request['version'] and not state.get('legacy'):
                        active_manager = cls(active.base)
                        records = _read(active_manager.manifest_file)
                        evaluation = _read(active.base / 'snapshot/evaluation_manifest.json')
                        healthy = bool(records) and evaluation.get('artifacts') == _digest(records) and all(
                            active_manager.healthy_record(key, record) for key, record in records.items())
                        if not healthy:
                            manager.save(accepted_request=None, status='needs_training',
                                         repair_generation=int(state.get('repair_generation', 0)) + 1)
                            request['repair_generation'] = int(state.get('repair_generation', 0)) + 1
                            request['version'] = _digest({k: v for k, v in request.items() if k not in ('version', 'end')})
                            requests.append(request)
                    elif state.get('accepted_request') != request['version']:
                        requests.append(request)
            except Exception as exc:
                Logger.warning(f'Lifecycle check failed for {registration["name"]}: {exc}')
        return requests

    def portable_files(self) -> list[Path]:
        """Include approval aliases and provenance in the existing archive transport."""
        from .core import ModelPath
        roots = [self.base]
        if origin := self.state.get('origin_model'):
            roots.append(ModelPath(origin).base)
        files = []
        for root in roots:
            files.extend(root.joinpath('configs').glob('*.yaml'))
            files.extend(path for path in (
                root / 'snapshot/model_manifest.json', root / 'snapshot/evaluation_manifest.json',
                root / 'logs/lifecycle.jsonl') if path.is_file())
        return sorted(set(files))

    def inspect(self) -> dict[str, Any]:
        """Return portable state, full target and available node provenance."""
        return {'state': self.state, 'production': _read(self.target_file),
                'artifacts': _read(self.manifest_file)}

    def configure_budget(self, *, enabled: bool = True, max_epoch: int = 20,
                         window: int = 1200, step: int = 2, mode: str = 'quick') -> None:
        self.training_options(mode, {'max_epoch': max_epoch, 'window': window, 'step': step})
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (max_epoch, window, step)):
            raise ValueError('Budget values must be positive integers')
        with self.lock():
            if self.state.get('stage') != 'exploration' or self.state.get('short_test'):
                raise ValueError('Only ordinary exploration models accept quick budgets')
            if mode == 'auto':
                raise ValueError('configure_budget requires an explicit quick, sparse or full mode')
            budget = {'enabled': enabled and mode != 'full', 'mode': mode if enabled else 'full',
                      'max_epoch': max_epoch, 'window': window, 'step': step}
            if budget != self.state.get('budget'):
                self.invalidate_evaluation()
                self.save(budget=budget, status='needs_training')

    def update_target(self, target: Mapping, *, retrain: bool = False) -> None:
        """Explicitly replace the full target; never implicitly follow a live schedule."""
        if not all(key in target for key in ('model', 'algo')):
            raise ValueError('Production target requires model and algo mappings')
        with self.lock():
            _write(self.target_file, target)
            self.save(target_version=_digest(target), status='needs_training', accepted_request=None,
                      repair_generation=int(self.state.get('repair_generation', 0)) + 1, force_retrain=retrain)

    def repair(self, request: Mapping, run: Callable[..., Any]) -> None:
        """Build an isolated existing-index run, then atomically publish its reference."""
        from src.res.model.util.core import ModelPath
        target = _read(self.target_file)
        if _digest(target) != request['target_version']:
            raise ValueError('Lifecycle target changed; repair request is stale')
        with self.lock():
            state = self.state
            if state.get('accepted_request') == request['version']:
                return
            source = (ModelPath(state['validated_model']) if state.get('validated_model')
                      else ModelPath(state['approved_model']) if state.get('approved_model')
                      else ModelPath(Path(state.get('validated_path') or state.get('approved_path') or str(self.base))))
            previous_candidate = state.get('candidate_path')
            staged = ModelLifecycle(Path(previous_candidate)) if previous_candidate else None
            reuse = bool(staged and staged.state.get('request_version') == request['version'])
            if reuse and staged is not None:
                candidate = ModelPath(staged.base)
                # Retry training preserves completed weights; evaluation always starts fresh.
                for folder in ('pred_recorder', 'basic_test', 'detailed_alpha'):
                    shutil.rmtree(candidate.snapshot(folder), ignore_errors=True)
                shutil.rmtree(candidate.rslt(), ignore_errors=True)
            else:
                candidate = ModelPath(self.base).with_new_index(ModelPath(self.base).find_new_index(folder_not_exist=True))
                candidate.base.mkdir(parents=True, exist_ok=False)
                candidate.conf().mkdir()
                _write(candidate.conf_file('model'), target['model'])
                _write(candidate.conf_file('schedule'), {})
                _write(candidate.conf_file(f'algo.{candidate.model_module}'), target['algo'])
                if target.get('boost_head'):
                    # Preserve the algorithm file name of the original boost head.
                    for file in self.base.joinpath('configs').glob('algo.*.yaml'):
                        if file.name != f'algo.{candidate.model_module}.yaml':
                            _write(candidate.conf() / file.name, target['boost_head'])
                staged = ModelLifecycle(candidate.base)
                _write(staged.target_file, target)
                staged.save(schema_version=1, stage='validation', status='needs_training', legacy=False,
                            target_version=request['target_version'], budget={'enabled': False},
                            origin=str(self.base), origin_model=ModelPath(self.base).full_name, request_version=request['version'])
                self.save(status='repairing', candidate_path=str(candidate.base))
            assert staged is not None
            # Only independent, matching nodes can be copied. Unknown dependencies retrain.
            model = target['model']
            input_types = (model.get('input.combo.types', []) if model.get('input.type') == 'combo'
                           else [model.get('input.type')])
            dependent = bool(state.get('force_retrain') or model.get('train.trainer.transfer')
                             or set(input_types).intersection(('hidden', 'final')))
            if not dependent and not reuse:
                source_manager = ModelLifecycle(source.base)
                manifest = _read(source_manager.manifest_file)
                if source_manager.state.get('legacy'):
                    from .config import ModelConfig
                    saved = ModelConfig(source, stage=2, resume=1, selection=0, short_test=False, vb_level='never')
                    if self.semantic(self.snapshot(saved)) == self.semantic(target):
                        for number in saved.model_num_list:
                            for date in request['nodes']:
                                for submodel in saved.submodels:
                                    key = f'{number}:{date}:{submodel}'
                                    weights = source_manager.weights(int(number), int(date), submodel)
                                    if self.required_weights(saved).issubset(weights):
                                        manifest.setdefault(key, {'definition': _digest(self.semantic(target)),
                                                                 'weights': weights, 'legacy': True})
                copied = {}
                for key, record in manifest.items():
                    number, date, submodel = key.split(':')
                    if int(date) not in request['nodes']:
                        continue
                    # Final compatibility is checked after production callbacks run.
                    # Incompatible copied nodes are retrained inside this isolated candidate.
                    weights = source_manager.weights(int(number), int(date), submodel)
                    if not weights or weights != record.get('weights'):
                        continue
                    shutil.copytree(source.full_path(number, date, submodel),
                                    candidate.full_path(number, date, submodel))
                    copied[key] = record
                _write(staged.manifest_file, copied)
            # A new candidate can legitimately contain zero artifacts; lifecycle repair
            # explicitly resumes its immutable configuration, not a failed ordinary resume.
            try:
                trainer = run(base_path=candidate, stage=0, resume=1, selection=0, short_test=False,
                              start=request.get('training_start', request['start']), end=request['end'], lifecycle_repair=True)
                from src.proj import Save
                Save.async_wait_all(caller_name='ModelLifecycle.repair')
                expected = [(int(d), int(n)) for d in trainer.data.model_date_list for n in trainer.config.model_num_list]
                if not expected or any(not staged.compatible(trainer.config, n, d) for d, n in expected):
                    raise ValueError('Production artifact coverage is incomplete')
                self._validate_predictions(candidate, trainer, expected)
                if _digest(_read(self.target_file)) != request['target_version']:
                    raise ValueError('Target changed during repair')
                if self.dependencies(trainer.config, production=True) != request.get('dependencies', {}):
                    raise ValueError('Upstream dependencies changed during repair')
                current = next((r for r in self.registrations() if r['base'] == str(self.base)), None)
                if (current is None or current['stage'] != request['stage']
                        or _digest(current) != request.get('registration_version', _digest(current))):
                    raise ValueError('Lifecycle registration changed during repair')
                evaluation = {'request': dict(request), 'artifacts': _digest(_read(staged.manifest_file)),
                              'status': 'passed'}
                _write(candidate.snapshot('evaluation_manifest.json'), evaluation)
                staged.save(stage=request['stage'], status='ready', accepted_request=request['version'])
                self.save(stage=request['stage'], status='ready', legacy=False, force_retrain=False,
                          accepted_request=request['version'],
                          approved_path=str(candidate.base) if request['stage'] == 'production' else state.get('approved_path'),
                          validated_path=str(candidate.base), validated_model=candidate.full_name,
                          approved_model=candidate.full_name if request['stage'] == 'production' else state.get('approved_model'),
                          previous_approved_path=state.get('approved_path'), previous_approved_model=state.get('approved_model'))
            except BaseException as exc:
                self.save(status='failed', candidate_path=str(candidate.base), error=f'{type(exc).__name__}: {exc}')
                raise

    @staticmethod
    def _validate_predictions(candidate, trainer, expected: list[tuple[int, int]]) -> None:
        from src.proj import Load
        import numpy as np
        folder = candidate.snapshot('pred_recorder', 'preds')
        frames = [Load.df(file) for file in folder.glob('*.feather')]
        if not frames:
            raise ValueError('No production evaluation predictions')
        import pandas as pd
        frame = pd.concat(frames, ignore_index=True)
        if frame.empty or not np.isfinite(frame['pred'].to_numpy()).all():
            raise ValueError('Production predictions are empty or non-finite')
        dates = set(int(d) for d in frame['date'].unique())
        if not set(int(d) for d in trainer.data.test_full_dates).issubset(dates):
            raise ValueError('Production prediction dates are incomplete')
        for date, number in expected:
            required_dates = {int(d) for d in trainer.data.test_full_dates
                              if date < int(d) <= trainer.data.next_model_date(date)}
            for submodel in trainer.config.submodels:
                subset = frame[(frame['model_date'] == date) & (frame['model_num'] == number)
                               & (frame['submodel'] == submodel)]
                if not required_dates.issubset(set(int(d) for d in subset['date'].unique())):
                    raise ValueError(f'Missing evaluated model dates: {number}/{date}/{submodel}')
