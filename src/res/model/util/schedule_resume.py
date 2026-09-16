"""Resolve unrecorded schedule runs before reading any current model configuration."""
from __future__ import annotations

import re
import sys
from collections.abc import Mapping

import yaml

from src.proj import MACHINE, PATH
from src.res.model.util.core import ModelPath

__all__ = ['find_schedule_runs', 'select_schedule_run', 'validate_saved_configs']


def find_schedule_runs(schedule_name: str, *, short_test: bool | None = None) -> list[ModelPath]:
    """Match the schedule across modules and indices, including incomplete folders."""
    if short_test is None:
        short_test = not MACHINE.cuda_server
    roots = [PATH.model_st] if short_test else [PATH.model_nn, PATH.model_boost]
    suffix = re.compile(r'@' + re.escape(schedule_name) + r'(?:@\d+)?$')
    candidates = []
    for root in roots:
        for path in root.glob('*'):
            if not path.is_dir() or not suffix.search(path.name):
                continue
            try:
                model = ModelPath(path)
            except (ValueError, AssertionError):
                continue
            if model.model_clean_name == schedule_name:
                candidates.append(model)
    return sorted(candidates, key=lambda model: (model.model_name_index, model.full_name))


def _creation_key(model: ModelPath):
    entries = [entry.timestamp.timestamp() for entry in model.log_file.read()
               if entry.title == 'create_model_path']
    path = model.conf_file('model')
    timestamp = max(entries) if entries else (path if path.exists() else model.base).stat().st_mtime
    return timestamp, model.model_name_index, model.full_name


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
        return max(candidates, key=_creation_key)
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
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f'Cannot resume: saved configuration is corrupt or unreadable: {path}') from exc
