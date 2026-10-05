"""Read only the source reports needed by the requested templates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import recover_returns
from .types import CompareConfig, CompareModelSpec


@dataclass
class ModelInput:
    name: str
    excel: Path
    inference_dir: Path | None
    hidden_model_num: int
    ic: pd.Series | None = None
    top: pd.DataFrame | None = None


def resolve_spec(value) -> ModelInput:
    spec = value if isinstance(value, CompareModelSpec) else CompareModelSpec(value)
    source = getattr(spec.model, 'base', spec.model)
    path = Path(source)
    directory = None
    if path.suffix.lower() == '.xlsx':
        excel = path
    else:
        if not path.is_dir():
            from src.res.model.util import ModelPath
            path = ModelPath(spec.model).base
        directory = path.resolve()
        excel = directory / 'results' / 'detailed_alpha_data.xlsx'
    if spec.excel_path is not None:
        excel = Path(spec.excel_path)
    if spec.inference_dir is not None:
        directory = Path(spec.inference_dir).resolve()
    if not excel.is_file():
        raise FileNotFoundError(excel)
    if spec.hidden_model_num < 0:
        raise ValueError('hidden_model_num must be nonnegative')
    return ModelInput(spec.name or (directory.name if directory else excel.stem.removeprefix('detailed_alpha_data_').split('_at_')[0]),
                      excel.resolve(), directory, spec.hidden_model_num)


def read_sheet(book: pd.ExcelFile, sheet: str, selectors: dict, date_col: str, metrics: list[str]):
    if sheet not in book.sheet_names:
        raise ValueError(f'Missing {sheet}; available: {book.sheet_names}')
    df = pd.read_excel(book, sheet_name=sheet)
    required = set(selectors) | {date_col} | set(metrics)
    if missing := required - set(df.columns):
        raise ValueError(f'{sheet}: missing columns {sorted(missing)}')
    # Only identity/index columns can contain merged-cell continuations.
    identity = [c for c in ('prefix', 'factor_name', 'benchmark', 'strategy', 'suffix', 'topN') if c in df]
    df[identity] = df[identity].ffill()
    available = df[list(selectors)].drop_duplicates().to_dict('records')
    for column, value in selectors.items():
        df = df.loc[df[column] == value]
    if df.empty:
        raise ValueError(f'{sheet}: no match for {selectors}; available: {available}')
    dates = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='raise')
    df = df.assign(**{date_col: dates.dt.strftime('%Y%m%d').astype(int)})
    if df.duplicated(date_col).any():
        raise ValueError(f'{sheet}: duplicate business key/date after selection {selectors}')
    df = df.set_index(date_col).sort_index()[metrics].apply(pd.to_numeric, errors='raise')
    return df.replace([np.inf, -np.inf], np.nan)


def load_inputs(models, config: CompareConfig):
    inputs = [resolve_spec(model) for model in models]
    if len(inputs) < 2:
        raise ValueError('Select at least two models')
    if len({m.name for m in inputs}) != len(inputs):
        raise ValueError('Model display names must be unique; set CompareModelSpec.name')
    if len({m.excel for m in inputs}) != len(inputs):
        raise ValueError('Select distinct model reports')
    needs_ic = bool(set(config.templates) & {'ic', 'complementarity'})
    needs_top = bool(set(config.templates) & {'top', 'complementarity'})
    for model in inputs:
        with pd.ExcelFile(model.excel) as book:
            if needs_ic:
                model.ic = read_sheet(book, 'factor@ic_curve',
                                      {'factor_name': config.submodel, 'benchmark': config.ic_benchmark},
                                      'date', ['ic']).ic
            if needs_top:
                curve = read_sheet(book, config.top_sheet,
                                   {'factor_name': config.submodel, 'benchmark': config.top_benchmark,
                                    'strategy': config.top_strategy, 'suffix': config.top_suffix,
                                    'topN': config.top_n}, 'trade_date', ['pf', 'bm', 'excess'])
                calendar = config.trade_dates
                if calendar is None:
                    from src.proj import CALENDAR
                    calendar = CALENDAR.range(int(curve.index.min()), int(curve.index.max()),
                                              'td', until_today=False)
                model.top = recover_returns(curve, calendar)
    return inputs


def common_dates(series, config: CompareConfig) -> pd.Index:
    dates = series[0].index
    for item in series[1:]:
        dates = dates.intersection(item.index)
    dates = dates.sort_values()
    if config.start is not None:
        dates = dates[dates >= config.start]
    if config.end is not None:
        dates = dates[dates <= config.end]
    if dates.empty:
        raise ValueError('No common dates within the requested window')
    return dates
