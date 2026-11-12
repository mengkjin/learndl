"""Resolve schedule resumes from current directories and effective training definitions."""
from __future__ import annotations

import re
import sys
from collections.abc import Mapping

import yaml

from src.proj import MACHINE, PATH
from src.res.model.util.core import ModelPath

__all__ = ['find_schedule_runs', 'select_schedule_run', 'validate_saved_configs',
           'training_definition', 'automatic_schedule_config']


def find_schedule_runs(schedule_name: str, *, short_test: bool | None = None) -> list[ModelPath]:
    """Match the schedule across modules and indices, including incomplete folders."""
    if short_test is None:
        short_test = not MACHINE.cuda_server
    roots = [PATH.model_st] if short_test else [PATH.model_nn, PATH.model_boost]
    suffix = re.compile(r'@' + re.escape(schedule_name) + r'(?:@(?P<index>\d+))?$')
    candidates = []
    for root in roots:
        for path in root.glob('*'):
            match = suffix.search(path.name)
            if (not path.is_dir() or not match
                    or (match['index'] is not None and int(match['index']) < 2)):
                continue
            try:
                model = ModelPath(path)
            except (ValueError, AssertionError):
                continue
            if model.model_clean_name == schedule_name:
                candidates.append(model)
    return sorted(candidates, key=lambda model: (model.model_name_index, model.full_name))


def select_schedule_run(schedule_name: str, *, short_test: bool | None = None,
                        policy: str = 'interactive') -> ModelPath:
    if policy not in ('interactive', 'latest'):
        raise ValueError(f'Unknown resume selection policy: {policy}')
    candidates = find_schedule_runs(schedule_name, short_test=short_test)
    if not candidates:
        raise FileNotFoundError(f'Cannot resume [{schedule_name}]: no existing training directory')
    if len(candidates) == 1:
        return candidates[0]
    if policy == 'latest':
        return max(candidates, key=lambda model: (model.model_name_index, model.full_name))
    labels = {str(model.base): model for model in candidates}
    if sys.stdin is None or not sys.stdin.isatty():
        raise ValueError(
            f'Multiple training directories for [{schedule_name}]; select one interactively '
            f'or pass an explicit base_path: {", ".join(labels)}'
        )
    from src.proj.util.cli import AskFor
    flag = AskFor.Options(
        list(labels), confirm=True, multiple=False,
        title=f'Choose the existing training directory for [{schedule_name}]',
        help_description='Resume the saved configuration in the selected directory. No directory is chosen by age.',
    )
    if not flag.valid or flag.result not in labels:
        raise InterruptedError(f'Resume directory selection cancelled for [{schedule_name}]')
    return labels[flag.result]


def validate_saved_configs(model: ModelPath, *, boost_head: str | None = None) -> None:
    """Missing saved configuration must not silently fall back to today's defaults."""
    from .config.config import get_config_dict
    keys = ['model', 'schedule', f'algo.{model.model_module}']
    if boost_head:
        keys.append(f'algo.{boost_head}')
    for key in keys:
        path = model.conf_file(key)
        if not path.is_file():
            raise FileNotFoundError(f'Cannot resume: saved configuration is missing: {path}')
        try:
            value = yaml.safe_load(path.read_text())
            if not isinstance(value, Mapping):
                raise ValueError('expected a YAML mapping')
            if key in ('model', 'schedule'):
                fields = get_config_dict(dict(value))
                module = fields.get('model.module')
                if key == 'model' and module is None:
                    raise ValueError('saved model.module is missing')
                if module is not None:
                    module = str(module).lower().replace('/', '@')
                    if module in ('nn', 'boost'):
                        module = fields.get(f'model.module.{module}')
                    if module not in (model.model_module, model.full_module_name):
                        raise ValueError(f'saved module {module!r} does not match {model.full_module_name}')
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f'Cannot resume: saved configuration is corrupt or unreadable: {path}') from exc


def training_definition(config) -> dict:
    """Canonical saved training definition, excluding operational settings."""
    from .config.config import BaseModelConfig

    # Old model.yaml files may predate optional fields. Use only the existing
    # project's compatibility defaults; never overlay today's schedule on them.
    values = dict(BaseModelConfig.OPTIONAL_CONFIG_PARAM) | dict(config.model_config.Param)
    excluded = ('model.name', 'train.callbacks', 'train.experimental_metrics_fastpath')
    definition = {
        key: value for key, value in values.items()
        if (key.startswith(('model.', 'input.', 'train.'))
            or key in ('env.precision', 'env.random_seed', 'env.allow_tf32'))
        and key not in excluded and not key.startswith('train.torch_compile')
    }
    definition['algo'] = config.algo_config.params
    definition['boost_head_algo'] = (config.boost_head_config.params
                                     if config.boost_head_config else None)
    return definition


def automatic_schedule_config(config_kwargs: dict, model_kwargs: dict):
    """Select the highest matching on-disk run, or prepare a safe fresh run.

    This function does not create directories. Existing paths are never cleared.
    Candidate selection is rechecked once before returning to start_model().
    """
    from .config import ModelConfig
    from .resume_validation import validate_resume

    schedule_name = config_kwargs['schedule_name']
    options = dict(model_kwargs)
    options.pop('resume_selection', None)
    options.update(resume=0, selection=0)
    target = ModelConfig(**(config_kwargs | {'base_path': None}), **options)
    expected = training_definition(target)
    logger = target.logger

    def snapshot(path):
        return {file.name: file.read_bytes() for file in path.conf().glob('*.yaml')}

    for _ in range(2):
        candidates = sorted(find_schedule_runs(schedule_name, short_test=target.short_test),
                            key=lambda p: (p.model_name_index, p.full_name), reverse=True)
        logger.note(f'Automatic resume: schedule={schedule_name}, requested_resume=True, '
                    f'candidates={[str(p.base) for p in candidates]}')
        for path in candidates:
            from .lifecycle import ModelLifecycle
            if ModelLifecycle(path.base).state.get('origin'):
                continue  # Isolated production candidates are not exploration resumes.
            try:
                before = snapshot(path)
                validate_saved_configs(path)
                saved = ModelConfig(path, stage=options.get('stage', 0), resume=1, selection=0,
                                    override={'env.short_test': target.short_test}, vb_level='never')
                actual = training_definition(saved)
                differences = sorted(key for key in expected.keys() | actual.keys()
                                     if expected.get(key) != actual.get(key))
                if differences:
                    logger.note(f'Automatic resume: skip {path.base}; definition differs: {differences}')
                    continue
                if saved.boost_head_config:
                    validate_saved_configs(path, boost_head=saved.boost_head_config.model_module)
                # Validate archived artifacts without invoking start_model, which writes.
                saved.is_resuming = True
                saved.queue_of_stages = ['fit'] if options.get('stage', 0) != 2 else ['test']
                validate_resume(saved)
                chosen = ModelConfig(**(config_kwargs | {'base_path': path}),
                                     **(options | {'resume': 1, 'short_test': path.is_short_test}))
                # A renamed directory's old model.name must not redirect selection.
                if chosen.base_path.base != path.base:
                    raise ValueError('saved configuration redirected the selected directory')
                if (not path.base.is_dir() or before != snapshot(path)
                        or {p.base for p in candidates} != {p.base for p in find_schedule_runs(
                            schedule_name, short_test=target.short_test)}):
                    break
                logger.note(f'Automatic resume: selected={path.base}; final_resume=True')
                return chosen
            except (OSError, ValueError, AssertionError, KeyError, TypeError) as exc:
                if not path.base.is_dir():
                    break
                logger.note(f'Automatic resume: skip {path.base}; {type(exc).__name__}: {exc}')
        else:
            if {p.base for p in candidates} != {
                    p.base for p in find_schedule_runs(schedule_name, short_test=target.short_test)}:
                logger.note('Automatic resume: directory list changed; rescanning')
                continue
            # Use a new suffix above every occupied candidate, including invalid
            # directories, rather than allowing start_model to clear an old run.
            index = max([p.model_name_index for p in candidates] + [0]) + 1
            target.base_path.with_new_index(index)
            while target.base_path.base.exists():
                index += 1
                target.base_path.with_new_index(index)
            target.options.resume = 0
            target.options.selection = 0
            logger.note(f'Automatic resume: no compatible directory; '
                        f'new={target.base_path.base}; final_resume=False')
            return target
        logger.note('Automatic resume: candidate changed during validation; rescanning')
    raise RuntimeError('Automatic resume: directories/configuration changed repeatedly during selection')
