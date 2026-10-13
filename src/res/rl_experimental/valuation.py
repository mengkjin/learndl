"""Causal close valuations; tradeability always comes from raw day quotes."""
from dataclasses import dataclass

import numpy as np

from .progress import progress

REASONS = {0: 'unresolved', 1: 'traded_close', 2: 'nontrading_day_close',
           3: 'adjprice_fallback', 4: 'missing_data_carry', 5: 'not_listed'}


@dataclass
class Valuations:
    close: np.ndarray
    reason: np.ndarray
    stale_days: np.ndarray
    last_valid_date: np.ndarray
    stale: np.ndarray
    report: dict


def build_valuations(dates, stocks, raw_close, factors, status, adj_close, listed) -> Valuations:
    """Walk dates once. Never backfill, and reset historical anchors at delisting."""
    shape = raw_close.shape
    close = np.full(shape, np.nan)
    reason = np.zeros(shape, dtype=np.uint8)
    age = np.zeros(shape, dtype=np.int32)
    last_dates = np.zeros(shape, dtype=np.int32)
    last = np.full(shape[1], np.nan)
    last_date = np.zeros(shape[1], dtype=np.int32)
    duration = np.zeros(shape[1], dtype=np.int32)
    day_close = raw_close * factors
    day_ok = np.isfinite(day_close) & (day_close > 0) & np.isfinite(factors) & (factors > 0)
    adj_ok = np.isfinite(adj_close) & (adj_close > 0)
    conflicts = listed & day_ok & adj_ok & ~np.isclose(day_close, adj_close, rtol=1e-6, atol=1e-8)
    for t, date in enumerate(dates):
        alive = listed[t]
        last[~alive] = np.nan
        duration[~alive] = 0
        direct = alive & day_ok[t]
        fallback = alive & ~day_ok[t] & adj_ok[t]
        carry = alive & ~day_ok[t] & ~adj_ok[t] & np.isfinite(last)
        close[t, direct] = day_close[t, direct]
        close[t, fallback] = adj_close[t, fallback]
        close[t, carry] = last[carry]
        fresh = direct & (status[t] > 0)
        reason[t, ~alive] = 5
        reason[t, direct] = 2
        reason[t, fresh] = 1
        reason[t, fallback] = 3
        reason[t, carry] = 4
        valid = np.isfinite(close[t])
        observed = (direct | fallback) & (~np.isfinite(last) | ~np.isclose(close[t], last, rtol=1e-6, atol=1e-8) | fresh)
        last_date[observed] = int(date)
        duration[valid & ~fresh] += 1
        duration[fresh] = 0
        age[t] = duration
        last_dates[t] = last_date
        last[valid] = close[t, valid]
    stale = (reason != 1)

    def segments(mask):
        result = []
        for col in np.flatnonzero(mask.any(axis=0)):
            edges = np.diff(np.r_[False, mask[:, col], False].astype(np.int8))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
                result.append({'secid': int(stocks[col]), 'start': int(dates[start]),
                               'end': int(dates[end - 1]), 'days': int(end - start)})
        return result

    carry_segments = segments(reason == 4)
    conflict_segments = segments(conflicts)
    unresolved = np.argwhere((reason == 0) & listed)
    report = {
        'reason_codes': REASONS,
        'reason_counts': {name: int((reason == code).sum()) for code, name in REASONS.items()},
        'reason_samples': {
            name: [{'date': int(dates[t]), 'secid': int(stocks[c])}
                   for t, c in np.argwhere(reason == code)[:10]]
            for code, name in REASONS.items()
        },
        'conflicting_price_cells': int(conflicts.sum()), 'conflict_intervals': conflict_segments,
        'missing_data_carry_intervals': carry_segments,
        'max_stale_days': int(age.max(initial=0)),
        'longest_stale_intervals': sorted(segments(stale & np.isfinite(close)),
                                         key=lambda item: item['days'], reverse=True)[:20],
        'unresolved_samples': [{'date': int(dates[t]), 'secid': int(stocks[c])} for t, c in unresolved[:20]],
        'age_note': 'Days since last real day close within loaded history; pre-window duration is unknown.',
    }
    for title, intervals in [('missing day/adjprice close: carrying valuation', carry_segments),
                             ('day/adjprice price conflict: using day close * factor', conflict_segments)]:
        if intervals:
            progress('valuation', f'{title}; {len(intervals)} security intervals; '
                     f'sample={intervals[:10]}; all intervals in quality report', warning=True)
    return Valuations(close, reason, age, last_dates, stale, report)


def valuation_context(panel, index: int) -> dict | None:
    if panel.valuation_reason is None:
        return None
    return {'reason': panel.valuation_reason[index], 'stale_days': panel.valuation_stale_days[index],
            'last_valid_date': panel.valuation_last_valid_date[index]}


def holding_valuation_metrics(panel, index: int, weights) -> dict:
    if panel.valuation_reason is None:
        return {}
    held = weights > 1e-8
    reason = panel.valuation_reason[index]
    return {'carried_holdings': int(np.count_nonzero(held & panel.valuation_stale[index])),
            'missing_data_carried_holdings': int(np.count_nonzero(held & (reason == 4))),
            'max_held_stale_days': int(panel.valuation_stale_days[index, held].max(initial=0))}
