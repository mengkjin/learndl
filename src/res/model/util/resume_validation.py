"""Validate stored training artifacts before resuming instead of silently starting over."""
from __future__ import annotations

from collections.abc import Mapping

from src.proj import Load

__all__ = ['validate_resume']


def validate_resume(config) -> None:
    if not config.is_resuming or 'fit' not in config.queue_of_stages or config.base_path.is_null_model:
        return
    base = config.base_path
    if not base.base.is_dir():
        raise FileNotFoundError(f'Cannot resume: training directory does not exist: {base.base}')
    if not base.conf_file('model').is_file():
        raise FileNotFoundError(f'Cannot resume: model configuration is missing: {base.conf_file("model")}')
    dates = sorted(path for path in base.archive().glob('*/*')
                   if path.is_dir() and path.name.isdigit() and path.parent.name.isdigit())
    if not dates:
        raise FileNotFoundError(f'Cannot resume: no saved models in {base.archive()}')
    keys = ['state_dict'] if config.module_type == 'nn' else ['boost_dict']
    if config.boost_head_config:
        keys.append('boost_head')
    for date in dates:
        for submodel in config.submodels:
            for key in keys:
                path = date / submodel / f'{key}.pt'
                if not path.is_file():
                    raise FileNotFoundError(f'Cannot resume: saved model is missing: {path}')
                try:
                    value = Load.torch(path, map_location='cpu')
                    if not isinstance(value, Mapping) or not value:
                        raise ValueError('expected a non-empty model dictionary')
                    del value
                except Exception as exc:
                    raise ValueError(f'Cannot resume: saved model is corrupt or unreadable: {path}') from exc
