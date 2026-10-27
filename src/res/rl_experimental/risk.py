"""Point-in-time, read-only adapter for the project's ST and warnst rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .progress import progress


def warning_ids(frame: pd.DataFrame, date: int) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=np.int64)
    start = int((pd.Timestamp(str(date)) - pd.Timedelta(days=89)).strftime('%Y%m%d'))
    dates = pd.to_numeric(frame['date'], errors='raise')
    matches = frame.title.str.contains(r'可能.*(?:风险警示)|可能.*(?:退市)', regex=True, na=False)
    return frame.loc[matches & dates.between(start, date), 'secid'].astype(int).unique()


def delisting_events(decisions, executions, stocks, description):
    events = np.zeros((len(decisions), len(stocks)), dtype=bool)
    positions = {int(s): i for i, s in enumerate(stocks)}
    for row in description.itertuples():
        col = positions.get(int(row.secid))
        if col is not None:
            events[:, col] |= (decisions < int(row.delist_dt)) & (executions >= int(row.delist_dt))
    return events


def prepare_risk(dates, stocks, db, info):
    """Load each announcement file once; never call the cache-writing get_warnst."""
    reason = np.zeros((len(dates), len(stocks)), dtype=np.uint8)
    first = pd.Timestamp(str(int(dates[0]))) - pd.Timedelta(days=89)
    calendar = [int(d.strftime('%Y%m%d')) for d in pd.date_range(first, str(int(dates[-1])))]
    available = {}
    file_counts = {}
    frames = []
    progress('risk', 'Reading available announcement files (read-only)')
    for exchange in ('bse', 'szse', 'sse'):
        key = f'announcement_{exchange}'
        present = set(int(d) for d in db.dates('crawler', key))
        available[exchange] = present
        counts = {'loaded': 0, 'empty': 0, 'empty_weekend': 0}
        file_counts[exchange] = counts
        for date in calendar:
            if date not in present:
                continue
            # Empty daily announcement files are normal, including weekends.
            # Keep reading all dates/exchanges; summarize instead of flooding logs.
            frame = db.load('crawler', key, date, vb_level='never')
            counts['loaded'] += 1
            if frame.empty:
                counts['empty'] += 1
                counts['empty_weekend'] += int(pd.Timestamp(str(date)).dayofweek >= 5)
                continue
            frame = frame.loc[frame.title.str.contains(r'可能.*(?:风险警示)|可能.*(?:退市)', regex=True, na=False)].copy()
            # Both publication date and file availability must precede the decision.
            frame['available_date'] = np.maximum(pd.to_numeric(frame['date']), date)
            frames.append(frame)
    announcements = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    progress('risk', 'Announcement files: ' + '; '.join(
        f'{ex}: loaded={c["loaded"]}, empty={c["empty"]} (weekend={c["empty_weekend"]})'
        for ex, c in file_counts.items()))
    coverage = []
    for i, date in enumerate(dates):
        date = int(date)
        if i % 128 == 0:
            progress('risk', f'ST/warning eligibility: {i + 1}/{len(dates)} dates')
        reason[i, np.isin(stocks, info.get_st(date).secid)] |= 1
        cache = db.path('exposure', 'warns_st_and_delist_90', date)
        if cache.exists():
            frame = db.load('exposure', 'warns_st_and_delist_90', date)
            if not frame.empty and 'date' in frame:
                start = int((pd.Timestamp(str(date)) - pd.Timedelta(days=89)).strftime('%Y%m%d'))
                frame = frame[pd.to_numeric(frame['date']).between(start, date)]
            ids = frame.secid.astype(int).unique() if not frame.empty else []
            coverage.append({'date': date, 'source': 'daily_cache', 'coverage': 'cache_origin_unverified'})
        else:
            frame = announcements if announcements.empty else announcements[announcements.available_date <= date]
            ids = warning_ids(frame, date)
            window = [int(d.strftime('%Y%m%d')) for d in pd.date_range(pd.Timestamp(str(date)) - pd.Timedelta(days=89), str(date))]
            missing = sum(d not in available[ex] for ex in available for d in window)
            coverage.append({'date': date, 'source': 'announcements', 'missing_exchange_days': missing,
                             'coverage': 'incomplete' if missing else 'complete'})
        reason[i, np.isin(stocks, ids)] |= 2
    incomplete = sum(c['coverage'] != 'complete' for c in coverage)
    report = {'rules': ['DATAVENDOR.INFO.get_st', 'warnst: 90 calendar days; 可能.*风险警示 or 可能.*退市'],
              'reason_bits': {'1': 'ST/abnormal', '2': 'announcement warning'},
              'st_cells': int(((reason & 1) != 0).sum()), 'warning_cells': int(((reason & 2) != 0).sum()),
              'unverified_or_incomplete_dates': incomplete, 'coverage': coverage,
              'announcement_file_counts': file_counts}
    progress('risk', f'Excluded ST cells={report["st_cells"]}, warning cells={report["warning_cells"]}; '
             f'incomplete/unverified warning coverage={incomplete}/{len(dates)}. '
             'Available rules applied; this is not a complete reproduction of live screening.', warning=bool(incomplete))
    return reason != 0, reason, report


def execution_risk_args(panel, index):
    return {'delisting_event': None if panel.delisting_event is None else panel.delisting_event[index],
            'risk_excluded': None if panel.risk_excluded is None else panel.risk_excluded[index]}


def execution_risk_metrics(panel, index, result, previous_nav):
    ids = [int(panel.stock_ids[i]) for i in result.written_off_indices]
    return {'written_off_stock_ids': ids, 'delisting_events': len(ids),
            'delisting_loss': previous_nav * result.delisting_loss_fraction,
            'assets_exhausted': result.assets_exhausted,
            'risk_unsold_holdings': int(np.count_nonzero((result.end_weights > 1e-8) & panel.risk_excluded[index]))
            if panel.risk_excluded is not None else 0}
