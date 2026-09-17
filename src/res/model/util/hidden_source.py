"""Exact hidden-source references; never discover or interactively select a model."""
from __future__ import annotations

import re

from src.res.algo import AlgoModule
from src.res.model.util.core import ModelPath, PredictorPath
from src.res.model.util.schedule_resume import validate_saved_configs

_REFERENCE = re.compile(
    r'(?:(st)@)?(?:(nn|boost)@)?([A-Za-z][A-Za-z0-9_]*)@'
    r'([A-Za-z0-9_][A-Za-z0-9_.-]*)(?:@([1-9][0-9]*))?@'
    r'(0|[1-9][0-9]*)@(best|swalast|swabest)'
)


def hidden_source_path(reference: str) -> PredictorPath:
    """module@name@directory_index@model_num@submodel; legacy omitted index is exactly 1."""
    match = _REFERENCE.fullmatch(reference) if isinstance(reference, str) else None
    if match is None:
        raise ValueError(f'Invalid hidden source {reference!r}; expected '
                         'module@model_name@directory_index@model_num@submodel '
                         '(directory_index >= 1, model_num >= 0, submodel best/swalast/swabest)')
    short_test, declared_type, module, name, index, number, submodel = match.groups()
    module_type = AlgoModule.module_type(module, raise_error=False)
    if module_type not in ('nn', 'boost') or (declared_type and declared_type != module_type):
        raise ValueError(f'Invalid or mismatched hidden model module: {reference}')
    model = ModelPath(f'{"st@" if short_test else ""}{module_type}@{module}@{name}')
    model.with_new_index(int(index or 1))
    if not model.base.is_dir():
        raise FileNotFoundError(f'Hidden source {reference}: training directory does not exist: {model.base}')
    validate_saved_configs(model)
    archive = model.archive(int(number))
    dates = sorted(path for path in archive.glob('*') if path.is_dir() and path.name.isdigit())
    if not dates:
        raise FileNotFoundError(f'Hidden source {reference}: model number {number} has no saved model dates: {archive}')
    checkpoint = 'state_dict.pt' if module_type == 'nn' else 'boost_dict.pt'
    for date in dates:
        path = date / submodel / checkpoint
        if not path.is_file():
            raise FileNotFoundError(f'Hidden source {reference}: checkpoint does not exist: {path}')
    return PredictorPath(model.base, int(number), submodel)
