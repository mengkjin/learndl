"""Model lifecycle policy, portable provenance and daily production training plans."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import local
from typing import Any
from collections.abc import Callable, Iterator, Mapping, Sequence

import portalocker
import yaml

from src.res.model.util.config.status_logger import rename_status_callback

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


def _target_digest(target: Mapping) -> str:
    # The logger rename is operational and must preserve existing approvals.
    return _digest(rename_status_callback(target, legacy=True))


class ModelLifecycle:
    """One model's policy. Runtime dictionaries contain serialized config values."""

    STAGES = ('exploration', 'validation', 'production')
    _locks = local()

    def __init__(self, base: Path):
        self.base = Path(base)
        self.file = self.base / 'configs/lifecycle.yaml'
        self.target_file = self.base / 'configs/production.yaml'
        self.manifest_file = self.base / 'snapshot/model_manifest.json'

    @property
    def state(self) -> dict[str, Any]:
        state = _read(self.file)
        if state.get('stage') == 'production' and not self.is_registered():
            return state | {'stage': 'exploration', 'budget': state.get('budget', {}) | {'enabled': False, 'mode': 'full'}}
        return state

    def is_registered(self) -> bool:
        return any(row['base'] == str(self.base) for row in self.registrations())

    @contextmanager
    def lock(self) -> Iterator[None]:
        # Daily repair and its synchronous trainer share the same model directory.
        owned = self._locks.__dict__.setdefault('owned', set())
        key = (os.getpid(), self.base.resolve())
        if key in owned:
            yield
            return
        self.base.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.base / '.lifecycle.lock'), timeout=0):
            owned.add(key)
            try:
                yield
            finally:
                owned.remove(key)

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
        return {'model': rename_status_callback(model, legacy=True), 'algo': snapshot['algo'], 'boost_head': snapshot.get('boost_head'),
                'dependencies': snapshot.get('dependencies', {})}

    @classmethod
    def validate_entry(cls, config) -> None:
        manager = cls(config.base_path.base)
        if config.lifecycle_promotion:
            state = manager.state
            if not (config.lifecycle_repair and config.is_resuming and not manager.is_registered()
                    and state.get('status') == 'upgrading'
                    and state.get('promotion_version') == config.lifecycle_promotion):
                raise ValueError('Lifecycle promotion requires its active, isolated candidate')
        elif config.lifecycle_repair:
            if not (config.is_resuming and manager.is_registered() and manager.state.get('status') == 'updating'):
                raise ValueError('Production resume requires an active registered model update')
        elif 'fit' in config.queue_of_stages:
            if manager.is_registered():
                raise ValueError('Use the production update to continue a registered model')
            if manager.state.get('promotion_version'):
                raise ValueError('Use a lifecycle promotion request to modify its isolated candidate')

    @classmethod
    def dump_training_config(cls, config, kwargs: dict) -> None:
        manager = cls(config.base_path.base)
        if manager.state:
            path = config.base_path.conf_file('model')
            if not path.exists() or kwargs.get('overwrite'):
                _write(path, rename_status_callback(_read(manager.target_file)['model']))
        else:
            config.model_config.Param.dump_yaml(config.base_path.conf_file('model'), **kwargs)

    @classmethod
    def training_nodes(cls, config, available_dates) -> list[int]:
        from src.proj import CALENDAR
        anchor = config.model_config.beg_date
        grid = CALENDAR.range(anchor, config.end_date, 'td')[::config.interval]
        nodes = cls.selected_nodes(config, grid.tolist())
        available = set(int(d) for d in available_dates)
        if 'fit' not in config.queue_of_stages:
            return [d for d in nodes if d in available]
        return [d for d in nodes if cls.fit_sample_cutoff(config, d) in available]

    @staticmethod
    def fit_sample_cutoff(config, model_date: int) -> int:
        """Last inclusive sample day, measured from the model day on the trading calendar."""
        from src.proj import CALENDAR
        return int(CALENDAR.td(model_date, -1 - config.skip_horizon))

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
                         legacy=legacy, short_test=config.short_test, target_version=_target_digest(target),
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
            if production:
                cls.approved_path(path, registered=True)
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
        if pending and trainer.config.transfer_training and trainer.config.lifecycle_repair:
            # A retry after a partial transfer chain must still refit downstream nodes.
            manifest = _read(manager.manifest_file)
            for date, number in pending:
                for submodel in trainer.config.submodels:
                    manifest.pop(f'{number}:{date}:{submodel}', None)
            _write(manager.manifest_file, manifest)
        return pending

    @classmethod
    def record_node(cls, trainer) -> None:
        manager = cls(trainer.base_path.base)
        if not manager.state:
            return
        from src.proj import Save
        Save.async_wait_all(raise_first_error=True, caller_name='ModelLifecycle.record_node')
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
        """Select the exact configured identity, including its optional directory index."""
        manager = cls(path.base)
        listed = manager.is_registered()
        if registered and not listed:
            raise ValueError(f'Model is not registered in prediction: {path.full_name}')
        state = manager.state
        if listed:
            updating = state.get('update_in_place') and state.get('status') in ('updating', 'failed')
            awaiting_acceptance = (registered and state and not state.get('legacy')
                                   and (state.get('stage') != 'production' or state.get('status') == 'needs_training'))
            if updating or awaiting_acceptance:
                raise ValueError(f'Production update has not passed acceptance: {path.full_name}')
        return path

    @classmethod
    def registrations(cls) -> list[dict[str, Any]]:
        from src.proj import Const
        from src.res.model.util.core import ModelPath
        strategy = Const.Model.strategies
        entries = {}
        for entry in strategy.get('prediction', {}).values():
            name = entry['name']
            path = ModelPath(name)
            if path.is_short_test:
                raise ValueError(f'Cannot register a short test for prediction: {name}')
            options = strategy.get('lifecycle', {}).get(name, {})
            entries[str(path.base)] = {'name': name, 'base': str(path.base), 'stage': 'production',
                                       **{key: options[key] for key in ('start', 'end') if key in options}}
        return list(entries.values())

    @classmethod
    def production_names(cls) -> set[str]:
        from .core import ModelPath
        return {ModelPath(row['name']).full_name for row in cls.registrations()}

    @staticmethod
    def catalog_stage(name: str, saved: str | None, registered: set[str], *, archived: bool):
        if name in registered and not archived:
            return 'production'
        return 'exploration' if saved == 'production' else saved

    def repair_plan(self, registration, state):
        from src.proj import CALENDAR
        from src.res.model.util.config import ModelConfig
        from src.res.model.util.core import ModelPath

        target = _read(self.target_file)
        model = target['model']
        start = int(registration.get('start', model['model.beg_date']))
        end = min(int(registration.get('end', model['model.end_date'])), int(CALENDAR.update_to()))
        target_version = _target_digest(target)
        dependency_config = ModelConfig(ModelPath(self.base), stage=2, resume=1, selection=0,
                                        override=model, short_test=False, vb_level='never')
        dependencies = self.dependencies(dependency_config, production=True)
        # Calendar cutoff changes only when a new training node becomes due.
        kind = ModelPath(self.base).module_type
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
        check_data_cutoff = not folds and not model.get('input.factor_sequence.dataset')
        return request, dependency_config, check_data_cutoff

    @classmethod
    def inspect_registered(cls, *, force_check: bool = False, on_error=None) -> list[dict[str, Any]]:
        """Reconcile the prediction registry against the calendar and accepted weights."""
        from src.proj import CALENDAR, Logger
        from src.res.model.util.config import ModelConfig
        from src.res.model.util.core import ModelPath
        requests = []
        for registration in cls.registrations():
            manager = cls(Path(registration['base']))
            try:
                active = ModelPath(registration['name'])
                if not active.requires_fit:
                    continue
                if not active.conf_file('model').is_file():
                    raise FileNotFoundError(f'Model configuration missing: {manager.base}')
                with manager.lock():
                    if not manager.file.exists():
                        config = ModelConfig(active, stage=2, resume=1, selection=0, short_test=False)
                        cls.prepare(config)
                    state = manager.state
                    if state.get('promotion_version') and (
                            state.get('promotion_stage') != 'production'
                            or state.get('status') in {'upgrading', 'upgrade_failed'}):
                        raise ValueError(f'Promotion has not passed production acceptance: {active.full_name}')
                    # Previous versions could point a prediction registration at another directory.
                    redirected = (state.get('approved_model') not in (None, active.full_name)
                                  if state.get('approved_model') else
                                  state.get('approved_path') not in (None, str(manager.base)))
                    if redirected:
                        manager.save(approved_model=None, approved_path=None, validated_model=None,
                                     validated_path=None, candidate_path=None, accepted_request=None,
                                     audit_version=None, legacy=bool(state.get('audit_note') or state.get('legacy')))
                    if state.get('stage') != 'production' or redirected:
                        manager.save(stage='production', target_stage='production', status='needs_training')
                    state = manager.state
                    request, dependency_config, check_data_cutoff = manager.repair_plan(registration, state)
                    grid = request['nodes']
                    if not grid:
                        continue
                    accepted = state.get('accepted_request') == request['version']
                    if state.get('legacy') and not state.get('force_retrain') and not state.get('target_changed'):
                        first_audit = not state.get('audit_version')
                        if first_audit:
                            manager.save(audit_version=request['version'], status='legacy',
                                         audit_note='Historical training evidence incomplete; existing artifacts retained')
                        if (first_audit or accepted) and not force_check:
                            weights = cls.required_weights(dependency_config)
                            files = (manager.file_stats(n, d, sub) for d in grid
                                     for n in dependency_config.model_num_list for sub in dependency_config.submodels)
                            if all(all(record.get(w, [0])[0] > 0 for w in weights) for record in files):
                                if first_audit:
                                    manager.save(accepted_request=request['version'])
                                continue
                        if accepted:
                            manager.save(accepted_request=None)
                    elif accepted:
                        records = _read(manager.manifest_file)
                        evaluation = _read(active.base / 'snapshot/evaluation_manifest.json')
                        required = {f'{n}:{d}:{s}' for d in grid for n in dependency_config.model_num_list
                                    for s in dependency_config.submodels}
                        if (required.issubset(records) and evaluation.get('artifacts') == _digest(records)
                                and all(manager.healthy_record(key, record) for key, record in records.items())):
                            continue
                        request['repair_generation'] += 1
                        manager.save(accepted_request=None, status='needs_training',
                                     repair_generation=request['repair_generation'])
                        request['version'] = _digest({k: v for k, v in request.items() if k not in ('version', 'end')})
                    if check_data_cutoff:
                        cutoff = cls.fit_sample_cutoff(dependency_config, grid[-1])
                        if CALENDAR.updated() < cutoff:
                            continue  # The both loader extends the historical dump through this date.
                    requests.append(request)
            except Exception as exc:
                Logger.warning(f'Lifecycle check failed for {registration["name"]}: {exc}')
                if on_error is not None:
                    on_error(registration['name'], exc)
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
            _write(self.target_file, rename_status_callback(target))
            self.save(target_version=_target_digest(target), status='needs_training', accepted_request=None,
                      repair_generation=int(self.state.get('repair_generation', 0)) + 1, force_retrain=retrain, target_changed=True)

    def check_registration(self, request: Mapping) -> None:
        current = next((r for r in self.registrations() if r['base'] == str(self.base)), None)
        if (current is None or current['stage'] != request['stage']
                or _digest(current) != request.get('registration_version', _digest(current))):
            raise ValueError('Lifecycle registration changed during repair')

    def repair(self, request: Mapping, run: Callable[..., Any]) -> None:
        """Resume in the prediction directory, preserving compatible weights and saved predictions."""
        from .core import ModelPath
        from .config import ModelConfig

        with self.lock():
            target = _read(self.target_file)
            if _target_digest(target) != request['target_version']:
                raise ValueError('Lifecycle target changed; repair request is stale')
            self.check_registration(request)
            state = self.state
            if state.get('accepted_request') == request['version']:
                return
            model = ModelPath(self.base)
            self.save(stage='production', status='updating', update_in_place=True, accepted_request=None)
            try:
                saved = ModelConfig(model, stage=2, resume=1, selection=0, short_test=False)
                matches_target = self.semantic(self.snapshot(saved)) == self.semantic(target)
                manifest = _read(self.manifest_file)
                if state.get('force_retrain'):
                    manifest = {}
                elif state.get('legacy') and matches_target and not request.get('dependencies'):
                    for number in saved.model_num_list:
                        for date in request['nodes']:
                            for submodel in saved.submodels:
                                key = f'{number}:{date}:{submodel}'
                                if key in manifest:
                                    continue
                                weights = self.weights(number, date, submodel)
                                if self.required_weights(saved).issubset(weights):
                                    manifest[key] = {'definition': _digest(self.semantic(target)),
                                                     'weights': weights, 'legacy': True,
                                                     'files': self.file_stats(number, date, submodel)}
                _write(self.manifest_file, manifest)
                if state.get('target_changed') or not matches_target:
                    # The production snapshot is fully resolved. Keep the
                    # original schedule for provenance before removing its overrides.
                    schedule = model.conf_file('schedule')
                    if schedule.is_file() and _read(schedule):
                        _write(model.snapshot(f'schedule_before_{request["target_version"]}.yaml'), _read(schedule))
                    _write(schedule, {})
                    _write(model.conf_file('model'), target['model'])
                    _write(model.conf_file(f'algo.{model.model_module}'), target['algo'])
                    if target.get('boost_head'):
                        head = target['model']['model.module.nn.boost_head.boost']
                        _write(model.conf_file(f'algo.{head}'), target['boost_head'])
                self.save(legacy=False, force_retrain=False, target_changed=False)
                trainer = self.fit_and_validate(
                    request, run, title=f'Updating {model.full_name}', log_operation='update_models')
                if _target_digest(_read(self.target_file)) != request['target_version']:
                    raise ValueError('Target changed during repair')
                if self.dependencies(trainer.config, production=True) != request.get('dependencies', {}):
                    raise ValueError('Upstream dependencies changed during repair')
                self.check_registration(request)
                evaluation = {'request': dict(request), 'artifacts': _digest(_read(self.manifest_file)),
                              'status': 'passed'}
                _write(model.snapshot('evaluation_manifest.json'), evaluation)
                self.save(status='ready', accepted_request=request['version'], approved_path=str(self.base),
                          approved_model=model.full_name, candidate_path=None, error=None)
            except BaseException as exc:
                self.save(status='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            from src.automation.artifact_exchange.production import publish_approved_model_safe
            publish_approved_model_safe(self.base)

    def fit_and_validate(self, request, run, *, title, **kwargs):
        from src.proj import Save
        from .core import ModelPath

        model = ModelPath(self.base)
        trainer = run(base_path=model, stage=0, resume=1, selection=0, short_test=False, use_data='both',
                      start=request.get('training_start', request['start']), end=request['end'],
                      title=title, lifecycle_repair=True, **kwargs)
        Save.async_wait_all(raise_first_error=True, caller_name='ModelLifecycle.fit_and_validate')
        requested_dates = set(map(int, request['nodes']))
        missing = requested_dates.difference(trainer.data.model_date_list)
        if missing:
            raise ValueError(f'Trainer omitted requested model dates: {sorted(missing)}')
        expected = [(d, int(n)) for d in sorted(requested_dates) for n in trainer.config.model_num_list]
        if not expected or any(not self.compatible(trainer.config, n, d) for d, n in expected):
            raise ValueError('Production artifact coverage is incomplete')
        try:
            self._validate_predictions(model, trainer, expected)
        except ValueError:
            # A partial saved evaluation can otherwise be skipped on every resume.
            self.invalidate_evaluation()
            raise
        return trainer

    @staticmethod
    def _validate_predictions(candidate, trainer, expected: list[tuple[int, int]]) -> None:
        from src.proj import Load
        import numpy as np
        if not len(trainer.data.test_full_dates):
            return  # No out-of-sample date exists for this first node yet.
        folder = candidate.snapshot('pred_recorder', 'preds')
        coverage: dict[tuple[int, int, str], set[int]] = {}
        for file in folder.glob('*.feather'):
            frame = Load.df(file)
            if frame.empty or not np.isfinite(frame['pred'].to_numpy()).all():
                raise ValueError('Production predictions are empty or non-finite')
            distinct = frame[['model_date', 'model_num', 'submodel', 'date']].drop_duplicates()
            for node, number, submodel, date in distinct.to_numpy():
                coverage.setdefault((int(node), int(number), str(submodel)), set()).add(int(date))
        if not coverage:
            raise ValueError('No production evaluation predictions')
        test_dates = set(map(int, trainer.data.test_full_dates))
        observed = set().union(*coverage.values())
        if not test_dates.issubset(observed):
            raise ValueError('Production prediction dates are incomplete')
        for date, number in expected:
            end = trainer.data.next_model_date(date)
            required_dates = {d for d in test_dates if date < d <= end}
            for submodel in trainer.config.submodels:
                if not required_dates.issubset(coverage.get((date, number, submodel), set())):
                    raise ValueError(f'Missing evaluated model dates: {number}/{date}/{submodel}')
