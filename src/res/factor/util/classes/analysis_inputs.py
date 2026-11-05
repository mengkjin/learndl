"""Factor analysis input contracts and cache scope."""

from __future__ import annotations

__all__ = [
    'Axis',
    'LabelParams',
    'EvaluationKind',
    'Quote',
    'LabelKey',
    'LABEL',
    'MarketInputs',
    'AnalysisInputs',
    'CURRENT',
    'return_label_scope',
    'evaluation_input',
    'return_labels',
]

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal, Protocol, TypeAlias

import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.proj import Base

Axis: TypeAlias = Sequence[int] | np.ndarray
LabelParams: TypeAlias = tuple[int, int, Base.lit.ReturnType]
EvaluationKind: TypeAlias = Literal["ret", "industry", "style"]
Quote: TypeAlias = tuple[np.ndarray, np.ndarray]
LabelKey: TypeAlias = tuple[LabelParams, tuple[int, ...], tuple[int, ...]]
LABEL: LabelParams = (10, 2, "close")


class MarketInputs(Protocol):
    def quote(self, date: int, end: int, price: str) -> Quote | None: ...
    def industry(self, date: int) -> pd.DataFrame | None: ...


class AnalysisInputs(MarketInputs, Protocol):
    def evaluation(
        self, kind: EvaluationKind, secids: Axis, dates: Axis, params: LabelParams
    ) -> pd.DataFrame | None: ...


CURRENT: ContextVar[AnalysisInputs | None] = ContextVar("analysis_inputs", default=None)
_RETURN_LABELS: ContextVar[dict[LabelKey, pd.DataFrame] | None] = ContextVar(
    "training_return_labels", default=None
)


@contextmanager
def return_label_scope() -> Iterator[None]:
    token = _RETURN_LABELS.set({})
    try:
        yield
    finally:
        _RETURN_LABELS.reset(token)


def evaluation_input(
    kind: EvaluationKind, secids: Axis, dates: Axis, params: LabelParams = LABEL
) -> pd.DataFrame | None:
    provider = CURRENT.get()
    return (
        None if provider is None else provider.evaluation(kind, secids, dates, params)
    )


def return_labels(secids: Axis, dates: Axis, params: LabelParams) -> pd.DataFrame:
    cache = _RETURN_LABELS.get()
    key = None
    if cache is not None:
        key = (params, tuple(map(int, secids)), tuple(map(int, dates)))
        if key in cache:
            return cache[key]
    labels = evaluation_input("ret", secids, dates, params)
    if labels is None:
        nday, lag, ret_type = params
        labels = DATAVENDOR.nday_fut_ret(
            np.asarray(secids), np.asarray(dates), nday, lag, ret_type=ret_type
        ).to_dataframe()
    if cache is not None and key is not None:
        cache[key] = labels
    return labels
