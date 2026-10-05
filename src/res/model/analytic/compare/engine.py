"""Run the three comparison templates over one normalized input collection."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .inputs import common_dates, load_inputs
from .statistics import correlation_matrix, ic_stats, periods, top_stats
from .types import CompareConfig, CompareResult


def compare_models(models, *, config=None, output_dir=None, display=True) -> CompareResult:
    config = config or CompareConfig()
    config.validate()
    inputs = load_inputs(list(models), config)
    result = CompareResult(summary=pd.DataFrame({'model': [m.name for m in inputs]}))
    diagnostics = []
    result.metadata = {'config': asdict(config), 'sources': [
        {'model': m.name, 'excel': str(m.excel),
         'inference_dir': str(m.inference_dir) if m.inference_dir else None,
         'hidden_model_num': m.hidden_model_num} for m in inputs],
        'methodology': {
            'ic': 'Raw IC; rolling window counts observations, not trading days.',
            'returns': 'pf/bm compound; excess accumulates daily pf-bm. Restore before date filtering.',
            'annualization': '365/calendar days; tracking error uses population standard deviation.',
            'missing': 'No imputation. Known endpoints give interval returns; gaps invalidate daily risk/annualized statistics.',
            'correlation': 'Finite pairs, at least 3 observations; mean output correlations equally weight dates.',
            'hidden': 'Centered linear CKA, no per-feature standardization; feature matrices never averaged over checkpoints.',
        }}
    candidate_dates = None
    for kind in ('ic', 'top'):
        series = [getattr(m, kind) for m in inputs]
        if series[0] is None:
            continue
        dates = common_dates(series, config)
        # Top is daily. With IC-only templates, use the calendar over the IC span.
        if kind == 'top' or candidate_dates is None:
            candidate_dates = dates
        spans = periods(dates, config)
        spans_with_all = {'all': dates, **spans}
        rows, daily, summary = [], [], []
        for model, source in zip(inputs, series):
            aligned = source.loc[dates].copy()
            base = {'model': model.name, 'start': int(dates.min()), 'end': int(dates.max())}
            for period, subset in spans_with_all.items():
                selected = aligned.loc[subset]
                stats = ic_stats(selected) if kind == 'ic' else top_stats(selected)
                row = {'model': model.name, 'period': period,
                       'start': int(subset.min()) if len(subset) else None,
                       'end': int(subset.max()) if len(subset) else None, **stats}
                if period in spans:
                    rows.append(row)
                if period == 'all':
                    summary.append({k if k == 'model' or k.startswith(f'{kind}_') else f'{kind}_{k}': v
                                    for k, v in row.items() if k != 'period'})
                if kind == 'top' and not stats['complete']:
                    diagnostics.append({'template': kind, 'model': model.name, 'period': period,
                                        'reason': 'Incomplete daily return path; annualization, IR and drawdown unavailable'})
            if kind == 'ic':
                df = aligned.to_frame('ic')
                df['rolling_ic'] = aligned.rolling(config.rolling_window, min_periods=config.rolling_window).mean()
                df['cum_ic'] = aligned.cumsum()
            else:
                df = aligned.copy()
                # Endpoints are known even where intervening daily observations are not.
                interrupted = np.r_[False, aligned.start.iloc[1:].to_numpy() != dates[:-1].to_numpy()]
                for col in ('pf', 'bm'):
                    baseline = df[f'previous_{col}'].iloc[0]
                    df[f'cum_{col}'] = (1 + df[f'source_{col}']) / (1 + baseline) - 1 if baseline != -1 else np.nan
                df['cum_excess'] = df.source_excess - df.previous_excess.iloc[0]
                if df.gap.iloc[0]:
                    df[['cum_pf', 'cum_bm', 'cum_excess']] = np.nan
                df['path_gap'] = interrupted | df.gap.to_numpy() | ~np.isfinite(df[['pf', 'bm', 'excess']]).all(axis=1)
                df['excess_drawdown'] = df.cum_excess - df.cum_excess.cummax().clip(lower=0)
                df.loc[df.path_gap.cummax(), 'excess_drawdown'] = np.nan
                for gap_date, gap_row in df.loc[df.path_gap].iterrows():
                    diagnostics.append({'template': 'top', 'model': model.name, 'date': int(gap_date),
                                        'reason': f'Unavailable daily return after {int(gap_row.start)}; endpoint retained'})
            df.index.name = 'date'
            daily.append(df.reset_index().assign(model=model.name))
            dropped = len(source.loc[(source.index >= dates.min()) & (source.index <= dates.max())]) - len(dates)
            if dropped:
                diagnostics.append({'template': kind, 'model': model.name,
                                    'reason': f'{dropped} dates excluded by common-date alignment'})
        result.tables[f'{kind}_periods'] = pd.DataFrame(rows)
        result.tables[f'{kind}_series'] = pd.concat(daily, ignore_index=True)
        result.summary = result.summary.merge(pd.DataFrame(summary), on='model', validate='1:1')
        result.metadata[f'{kind}_window'] = {**base, 'n_dates': len(dates)}
        result.metadata[f'{kind}_window'].pop('model')
        if 'complementarity' in config.templates:
            metrics = ['ic'] if kind == 'ic' else ['excess', 'pf']
            for metric in metrics:
                frame = pd.DataFrame({m.name: s.loc[dates] if kind == 'ic' else s.loc[dates, metric]
                                      for m, s in zip(inputs, series)})
                for period, subset in spans_with_all.items():
                    key = f'corr_{metric}_{period}'
                    matrix, counts, reasons = correlation_matrix(frame.loc[subset], config.corr_method)
                    result.correlations[key], result.correlations[f'{key}_count'] = matrix, counts
                    diagnostics.extend({'template': key, **reason} for reason in reasons)
    if config.analyze_pred or config.analyze_hidden:
        from .outputs import analyze_outputs
        if all(m.top is None for m in inputs):
            if config.trade_dates is not None:
                available = np.array(config.trade_dates)
                candidate_dates = available[(available >= candidate_dates.min()) & (available <= candidate_dates.max())]
            else:
                from src.proj import CALENDAR
                candidate_dates = CALENDAR.range(int(candidate_dates.min()), int(candidate_dates.max()),
                                                 'td', until_today=False)
        tables, matrices, samples, messages = analyze_outputs(inputs, config, list(candidate_dates))
        result.tables.update(tables)
        result.correlations.update(matrices)
        result.samples = samples
        diagnostics.extend(messages)
    result.diagnostics = pd.DataFrame(diagnostics)
    from .report import build_figures, make_highlights
    result.highlights = make_highlights(result)
    build_figures(result)
    result.export(output_dir)
    if display:
        result.display()
    return result
