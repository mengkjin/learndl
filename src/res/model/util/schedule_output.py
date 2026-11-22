"""Keep model output names stable while checking their schedule ownership."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import portalocker

from src.proj import PATH
from src.proj.env.schedules import find_schedule_path, schedule_matches, schedule_source


def _marker(model):
    return model.conf() / 'schedule_source.json'


def saved_reference(model) -> str | None:
    marker = _marker(model)
    return json.loads(marker.read_text())['reference'] if marker.is_file() else None


def linear_model_available(model) -> bool:
    path = find_schedule_path(saved_reference(model) or model.model_clean_name)
    return path is not None and path.is_relative_to(PATH.sched_linear)


def validate_model_source(model, source: Path) -> None:
    if model.model_clean_name != source.stem:
        raise ValueError(f'Model {model.base} does not belong to schedule {source}')
    marker = _marker(model)
    if marker.is_file():
        owner = json.loads(marker.read_text())['source']
        if owner != schedule_source(source):
            raise ValueError(f'Model {model.base} belongs to {owner}, not {schedule_source(source)}')
    elif model.base.is_dir() and len(schedule_matches(source.stem)) > 1:
        raise ValueError(f'Cannot identify the schedule that owns legacy model {model.base}: '
                         f'multiple configurations use {source.stem}')


@contextmanager
def guard_model_source(model, reference: str | None):
    source = find_schedule_path(reference)
    if source is None:
        yield None
        return
    directory = PATH.lc_machine / 'schedule_outputs'
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(source.stem.encode()).hexdigest()
    with portalocker.Lock(directory / f'{key}.lock', timeout=10):
        from .schedule_resume import find_schedule_runs
        from .core.model_path import ModelPath

        paths = {model.base: model}
        for short_test in (False, True):
            paths.update((p.base, p) for p in find_schedule_runs(source.stem, short_test=short_test))
        linear = PATH.model_linear / source.stem
        if linear.is_dir():
            paths[linear] = ModelPath(linear)
        for existing in paths.values():
            validate_model_source(existing, source)
        yield source


def save_model_source(model, source: Path) -> None:
    from .training_history import write_json

    identity = schedule_source(source)
    write_json(_marker(model), {'source': identity, 'reference': identity.split('/', 1)[1]})
