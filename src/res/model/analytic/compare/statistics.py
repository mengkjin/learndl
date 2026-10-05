"""Pure statistical operations; no inference, file writes or environment imports."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from .types import CompareConfig


def timestamp(date: int) -> pd.Timestamp:
    return pd.to_datetime(str(int(date)), format='%Y%m%d')


def sample_dates(dates, limit: int) -> list[int]:
    values = np.array(sorted(set(dates)), dtype=int)
    if limit < 1:
        raise ValueError('sample limit must be positive')
    if not len(values):
        return []
    positions = np.linspace(0, len(values) - 1, min(limit, len(values))).round().astype(int)
    return values[positions].tolist()


def periods(index, config: CompareConfig):
    dates = pd.Index(sorted(set(index)), dtype='int64')
    if dates.empty:
        return {}
    result = {}
    if 'all' in config.periods:
        result['all'] = dates
    ts = pd.to_datetime(dates.astype(str), format='%Y%m%d')
    for kind, freq in [('year', 'Y'), ('quarter', 'Q'), ('month', 'M')]:
        if kind in config.periods:
            labels = ts.to_period(freq).astype(str)
            for label in labels.unique():
                result[f'{kind}:{label}'] = dates[labels == label]
    if 'recent_year' in config.periods:
        cutoff = int((ts.max() - pd.DateOffset(years=1)).strftime('%Y%m%d'))
        result['recent_year'] = dates[dates > cutoff]
    for name, (start, end) in config.custom_periods.items():
        result[f'custom:{name}'] = dates[(dates >= start) & (dates <= end)]
    return result


def recover_returns(curve: pd.DataFrame, trade_dates) -> pd.DataFrame:
    """Recover before slicing. A missing trading session invalidates the next delta."""
    frame = curve.sort_index()
    if frame.index.has_duplicates:
        raise ValueError('Duplicate portfolio dates')
    calendar = pd.Index(sorted(set(trade_dates)))
    positions = calendar.get_indexer(frame.index)
    if (positions < 0).any():
        raise ValueError('Portfolio dates outside supplied trading calendar')
    result = pd.DataFrame(index=frame.index)
    for col in ('pf', 'bm'):
        previous = frame[col].shift(1)
        previous.iloc[0] = 0.0
        result[col] = (1 + frame[col]) / (1 + previous) - 1
        result[f'source_{col}'] = frame[col]
        result[f'previous_{col}'] = previous
    previous = frame.excess.shift(1)
    previous.iloc[0] = 0.0
    result['excess'] = frame.excess - previous
    result['source_excess'] = frame.excess
    result['previous_excess'] = previous
    gaps = np.r_[False, np.diff(positions) != 1]
    result.loc[gaps, ['pf', 'bm', 'excess']] = np.nan
    result['gap'] = gaps
    result['start'] = np.r_[frame.index[0], frame.index[:-1]]
    return result.replace([np.inf, -np.inf], np.nan)


def ic_stats(series: pd.Series) -> dict:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    return {'n': len(s), 'ic_mean': s.mean(), 'ic_std': s.std(ddof=1),
            'ic_positive_rate': (s > 0).mean() if len(s) else np.nan}


def top_stats(frame: pd.DataFrame) -> dict:
    """Match eval_pf_stats annualization, but include the zero drawdown baseline.

    Missing daily returns invalidate daily risk statistics. Interval endpoints
    still determine total return if the opening baseline is known.
    """
    keys = ('pf_return', 'bm_return', 'return_difference', 'excess_sum',
            'excess_annualized', 'tracking_error', 'ir', 'excess_mdd')
    valid = np.isfinite(frame[['pf', 'bm', 'excess']]).all(axis=1)
    result: dict[str, Any] = dict.fromkeys(keys, np.nan)
    continuous = len(frame) == 0 or np.array_equal(frame.start.iloc[1:].to_numpy(), frame.index[:-1].to_numpy())
    result.update(n=int(valid.sum()), complete=bool(len(frame) and valid.all() and continuous))
    if len(frame) and 'source_pf' in frame and not frame.gap.iloc[0]:
        with np.errstate(divide='ignore', invalid='ignore'):
            pf = float((1 + frame.source_pf.iloc[-1]) / (1 + frame.previous_pf.iloc[0]) - 1)
            bm = float((1 + frame.source_bm.iloc[-1]) / (1 + frame.previous_bm.iloc[0]) - 1)
        result.update(pf_return=pf if np.isfinite(pf) else np.nan,
                      bm_return=bm if np.isfinite(bm) else np.nan,
                      return_difference=pf - bm if np.isfinite(pf - bm) else np.nan,
                      excess_sum=float(frame.source_excess.iloc[-1] - frame.previous_excess.iloc[0]))
    if not result['complete']:
        return result
    pf = float(np.prod(1 + frame.pf.to_numpy(dtype=float)) - 1)
    bm = float(np.prod(1 + frame.bm.to_numpy(dtype=float)) - 1)
    x = frame.excess
    duration = (timestamp(frame.index.max()) - timestamp(frame.start.min())).days + 1
    growth = float(np.prod(1 + x.to_numpy(dtype=float)))
    ann = growth ** (365 / duration) - 1 if (x >= -1).all() and growth >= 0 else np.nan
    te = x.std(ddof=0) * np.sqrt(365 * len(x) / duration)
    cumulative = x.cumsum()
    dd = cumulative - cumulative.cummax().clip(lower=0)
    result.update(pf_return=pf, bm_return=bm, return_difference=pf - bm,
                  excess_sum=x.sum(), excess_annualized=ann, tracking_error=te,
                  ir=ann / te if te > 0 else np.nan, excess_mdd=-dd.min())
    return result


def correlation(x, y, method='pearson') -> tuple[float, int, str]:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        return np.nan, n, 'fewer than 3 paired observations'
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, n, 'constant values'
    if method == 'spearman':
        a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1]), n, ''


def correlation_matrix(frame, method='pearson'):
    columns = frame.columns
    values = pd.DataFrame(np.nan, index=columns, columns=columns)
    counts = pd.DataFrame(0, index=columns, columns=columns)
    reasons = []
    for i, a in enumerate(columns):
        for b in columns[i:]:
            value, count, reason = correlation(frame[a], frame[b], method)
            values.loc[a, b] = values.loc[b, a] = value
            counts.loc[a, b] = counts.loc[b, a] = count
            if reason:
                reasons.append({'model_a': a, 'model_b': b, 'reason': reason})
    return values, counts, reasons


def linear_cka(x, y) -> tuple[float, int, str]:
    """Biased linear CKA, centered over matched stocks; no feature standardization."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 3 or x.shape[1] == 0 or y.shape[1] == 0:
        return np.nan, n, 'insufficient representation observations'
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    # Scale first to avoid overflow without changing CKA.
    sx, sy = np.linalg.norm(x), np.linalg.norm(y)
    if sx == 0 or sy == 0:
        return np.nan, n, 'constant representation'
    x, y = x / sx, y / sy
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.clip(np.linalg.norm(x.T @ y) ** 2 / denominator, 0, 1)), n, ''
