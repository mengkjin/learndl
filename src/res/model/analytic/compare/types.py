"""Public configuration and result objects for reproducible model comparisons."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CompareModelSpec:
    """An Excel report, a training directory, or a ModelPath-compatible identifier.

    An explicit Excel path never implicitly selects a checkpoint directory.
    Set inference_dir to associate an exported historical report with its archives.
    """
    model: Any
    name: str | None = None
    excel_path: str | Path | None = None
    inference_dir: str | Path | None = None
    hidden_model_num: int = 0


@dataclass
class CompareConfig:
    start: int | None = None
    end: int | None = None
    periods: tuple[str, ...] = ('all', 'year', 'recent_year')
    custom_periods: dict[str, tuple[int, int]] = field(default_factory=dict)
    templates: tuple[str, ...] = ('ic', 'top', 'complementarity')
    submodel: str = 'best'
    ic_benchmark: str = 'market'
    top_sheet: str = 't50@perf_curve'
    top_strategy: str = 'Top_50'
    top_benchmark: str = 'univ'
    top_suffix: str = 'lag0'
    top_n: int = 50
    rolling_window: int = 20
    corr_method: str = 'pearson'
    analyze_pred: bool = False
    analyze_hidden: bool = False
    sample_num_corr: int = 20
    sample_num_hidden: int = 20
    # Optional explicit trading calendar, useful for portable/offline reports.
    trade_dates: tuple[int, ...] | None = None

    def validate(self):
        def date(value):
            return pd.to_datetime(str(value), format='%Y%m%d', errors='raise')
        for value in (self.start, self.end):
            if value is not None:
                date(value)
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError('start must not exceed end')
        if not self.periods or set(self.periods) - {'all', 'year', 'quarter', 'month', 'recent_year'}:
            raise ValueError('periods: choose all/year/quarter/month/recent_year')
        if not self.templates or set(self.templates) - {'ic', 'top', 'complementarity'}:
            raise ValueError('templates: choose ic/top/complementarity')
        if self.corr_method not in ('pearson', 'spearman'):
            raise ValueError('corr_method must be pearson or spearman')
        for key in ('rolling_window', 'sample_num_corr', 'sample_num_hidden', 'top_n'):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        for name, (start, end) in self.custom_periods.items():
            date(start)
            date(end)
            if not name or start > end:
                raise ValueError(f'Invalid custom period: {name}')


@dataclass
class CompareResult:
    summary: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    correlations: dict[str, pd.DataFrame] = field(default_factory=dict)
    samples: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)
    output_paths: dict[str, Path] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)

    def display(self):
        from src.proj import Logger
        columns = [c for c in ('model', 'ic_mean', 'top_pf_return', 'top_excess_sum',
                               'top_excess_annualized', 'top_ir', 'top_excess_mdd') if c in self.summary]
        Logger.display(self.summary[columns], title='Model comparison')
        for text in self.highlights:
            Logger.note(text)
        for key, figure in self.figures.items():
            if key in ('ic_trend', 'top_curves', 'corr_ic_all', 'corr_excess_all',
                       'pred_spearman_mean', 'hidden_cka_mean'):
                Logger.display(figure, title=key)
        if not self.diagnostics.empty:
            Logger.display(self.diagnostics, title='Comparison diagnostics')
        return self

    def export(self, output_dir: str | Path | None = None):
        from .report import export_report
        export_report(self, output_dir)
        return self
