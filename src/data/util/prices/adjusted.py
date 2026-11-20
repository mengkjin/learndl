"""
Adjusted-price panel with halt forward-fill.

For each trading day, store ``open/high/low/close/vwap`` already multiplied by
``adjfactor``.  Names missing from ``trade_ts/day`` (typical A-share halt)
are carried from the previous trading day's ``adjprice`` row. Unknown listing
metadata does not exclude a security; known pre-IPO and delisted names are removed,
so any two-date
period return is simply ``p1 / p0 - 1``.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import os
import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import fcntl
from uuid import uuid4

from src.proj import CALENDAR, DB, Save, Logger
from src.proj.db.interface.db_path import DBPath
from src.proj.db.io.snapshot import snapshot_active, snapshot_read_path

__all__ = ['calc_adjprice', 'ensure_adjprice', 'publish_day', 'covers_day', 'ensure_adjprices']

PRICE_COLS : tuple[str , ...] = ('open' , 'high' , 'low' , 'close' , 'vwap')


START_DATE = 19970101


@lru_cache(maxsize=1)
def first_date() -> int:
    return int(CALENDAR.range(START_DATE, START_DATE + 10000, until_today=False)[0])


def listing_description() -> pd.DataFrame:
    """Read listing intervals without initializing name-change/industry loaders."""
    description = DB.load('information_ts', 'description', vb_level='never')
    columns = ['secid', 'list_dt', 'delist_dt']
    if description is None or description.empty:
        return pd.DataFrame(columns=columns)
    if not set(columns).issubset(description.columns):
        raise ValueError('Invalid listing interval metadata')
    return description[columns]


def exclude_inactive(frame: pd.DataFrame, date: int) -> pd.DataFrame:
    """Unknown metadata is not an exclusion; known IPO/delisting bounds are."""
    desc = listing_description()
    # A code may have several listing intervals (e.g. 600018). Exclude it
    # only when every record is inactive, not when any single record is.
    outside = ((desc.list_dt > 0) & (desc.list_dt > int(date))) | \
              ((desc.delist_dt > 0) & (desc.delist_dt <= int(date)))
    active = desc.loc[~outside, 'secid']
    inactive = desc.loc[~desc.secid.isin(active), 'secid']
    return frame.loc[~frame.secid.isin(inactive)].copy()


@lru_cache(maxsize=4)
def _listing_intervals(path, version):
    return listing_description()


@lru_cache(maxsize=32768)
def _warn_unknown_listing(secid):
    Logger.warning(f'Portfolio listing status unknown for secid={secid}; retaining candidate')


def cached_listing_description() -> pd.DataFrame:
    path = Path(snapshot_read_path(DB.path('information_ts', 'description')))
    return _listing_intervals(path, tuple(stamp(path) or ()))


def rebalance_candidates(secids, model_date: int) -> np.ndarray:
    """Apply listing intervals on the next trading day, when the trade executes."""
    execution = int(CALENDAR.td(int(model_date), 1).as_int())
    desc = cached_listing_description()
    known = desc.loc[desc.list_dt.gt(0)]
    active = known.loc[known.list_dt.le(execution) &
                       (known.delist_dt.isna() | known.delist_dt.le(0) |
                        known.delist_dt.gt(execution)), 'secid']
    ids = np.asarray(secids)
    unknown = ~np.isin(ids, known.secid)
    for sid in ids[unknown]:
        _warn_unknown_listing(int(sid))
    return ids[unknown | np.isin(ids, active)]


def log_summary(reason, actions):
    if actions:
        counts = {kind: sum(action == kind for _, action in actions)
                  for kind in ('create', 'replace', 'unchanged')}
        Logger.stdout(f'adjprice trigger={reason} dates={min(d for d, _ in actions)}..'
                      f'{max(d for d, _ in actions)} processed={len(actions)} counts={counts}', vb_level=1)


def covers_day(adj: pd.DataFrame, day: pd.DataFrame, date: int) -> bool:
    """Check valid daily price cells, without requiring unavailable raw prices."""
    if day is None or day.empty or not {'secid', 'adjfactor'}.issubset(day.columns):
        return True
    expected = exclude_inactive(day, date).set_index('secid')
    if expected.empty:
        return True
    if adj is None or adj.empty or 'secid' not in adj.columns or adj.secid.duplicated().any():
        return False
    actual = adj.set_index('secid').reindex(expected.index)
    factor = expected.adjfactor.fillna(1.)
    for column in PRICE_COLS:
        if column not in expected.columns:
            continue
        raw = expected[column] * factor
        valid = np.isfinite(raw) & (raw > 0)
        if valid.any() and (column not in actual.columns or
                           not (np.isfinite(actual.loc[valid, column]) & (actual.loc[valid, column] > 0)).all()):
            return False
    return True



def calc_adjprice(date : int, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Adjusted OHLC/VWAP for the union of daily quotes and previous prices.

    - Traded names: ``price * adjfactor`` from ``trade_ts/day``.
    - Absent from day (halt): carry previous ``adjprice`` row.
    - Exclude only known pre-listing/delisted names; keep unknown metadata.
    - Still-all-NaN rows (e.g. IPO with no trade yet) are dropped.
    """
    date = int(date)
    day = DB.load('trade_ts' , 'day' , date , vb_level = 'never')
    required = {'secid', 'adjfactor', *PRICE_COLS}
    if day is None or day.empty or not required.issubset(day.columns):
        raise ValueError(f'Cannot calculate adjprice at {date}: trade_ts/day is missing, empty or lacks required columns')
    cols = ['secid' , *PRICE_COLS]
    day = exclude_inactive(day, date)
    if day.secid.duplicated().any():
        raise ValueError(f'Cannot calculate adjprice at {date}: duplicate daily securities')

    day = day.copy()
    adj = day['adjfactor'].fillna(1.0).to_numpy(float)
    traded = pd.DataFrame({'secid': day['secid'].to_numpy()})
    for c in PRICE_COLS:
        traded[c] = day[c].to_numpy(float) * adj
    traded = traded.set_index('secid')

    prev = CALENDAR.td(date, -1).as_int()
    if previous is None:
        previous = (DB.load('trade_ts', 'adjprice', prev, vb_level='never')
                    if prev >= CALENDAR.calendar_start() else pd.DataFrame())
    if (previous is None or previous.empty) and date != first_date():
        raise ValueError(f'Cannot calculate adjprice at {date}: previous trading day {prev} is missing')
    if previous is not None and not previous.empty:
        previous = exclude_inactive(previous, date)
        if previous.secid.duplicated().any():
            raise ValueError(f'Cannot calculate adjprice at {date}: duplicate previous securities')
        prior = previous.set_index('secid').reindex(columns=list(PRICE_COLS))
        result = traded.combine_first(prior)
    else:
        result = traded

    result = result.dropna(how = 'all' , subset = list(PRICE_COLS))
    return result.reset_index().loc[: , cols]


@contextmanager
def price_lock():
    """Serialize source publication and chained price writers across processes."""
    if snapshot_active():
        raise RuntimeError('Price publication is not allowed inside a read snapshot')
    path = DB.path('trade_ts', 'adjprice', first_date()).parent.parent / '.prices.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def stamp(path: Path) -> list[int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return [stat.st_size, stat.st_mtime_ns]


def dependencies(date: int) -> dict[str, list[int] | None]:
    previous = int(CALENDAR.td(date, -1).as_int())
    return {'day': stamp(DB.path('trade_ts', 'day', date)),
            'previous': None if date == first_date() else stamp(DB.path('trade_ts', 'adjprice', previous)),
            'description': stamp(DB.path('information_ts', 'description'))}


def fresh(date: int) -> bool:
    path = DB.path('trade_ts', 'adjprice', date)
    receipt = path.with_suffix('.source.json')
    try:
        record = json.loads(receipt.read_text())
        deps = dependencies(date)
        return (deps['day'] is not None and (date == first_date() or deps['previous'] is not None)
                and record == {'inputs': deps, 'output': stamp(path)} and record['output'] is not None)
    except (OSError, ValueError):
        return False


def atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.{uuid4().hex}.feather')
    try:
        Save.df(frame, temporary, vb_level='never')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_day(day: pd.DataFrame | None, date: int) -> None:
    if date < first_date() or int(CALENDAR.td(date).as_int()) != date:
        raise ValueError(f'Invalid day trading date {date}')
    if day is None or day.empty or not {'secid', 'adjfactor', *PRICE_COLS}.issubset(day.columns):
        raise ValueError(f'Cannot calculate adjprice at {date}: missing or invalid day quotes')
    for column in ('date', 'trade_date'):
        if column in day and not pd.to_numeric(day[column], errors='coerce').eq(date).all():
            raise ValueError(f'Cannot calculate adjprice at {date}: mismatched quote dates')
    if day.secid.isna().any() or day.secid.duplicated().any():
        raise ValueError(f'Cannot calculate adjprice at {date}: duplicate or missing securities')
    factor = day.adjfactor.fillna(1.).to_numpy(float)
    if not (np.isfinite(factor) & (factor > 0)).all():
        raise ValueError(f'Cannot calculate adjprice at {date}: invalid adjustment factor')
    values = day[list(PRICE_COLS)].to_numpy(float)
    with np.errstate(over='ignore', invalid='ignore'):
        adjusted = values * factor[:, None]
    if (not np.isfinite(values).any() or np.isinf(adjusted).any()
            or (values[np.isfinite(values)] <= 0).any()):
        raise ValueError(f'Cannot calculate adjprice at {date}: invalid prices')


def _ensure_adjprice(date: int, force: bool = False, reason: str = 'query',
                     verified: set[int] | None = None, actions: list | None = None) -> bool:
    boundary = first_date()
    if date < boundary or int(CALENDAR.td(date).as_int()) != date:
        raise ValueError(f'Invalid adjprice trading date {date}; first date is {boundary}')
    verified = set() if verified is None else verified
    pending = []
    cursor = date
    while True:
        if cursor in verified and not (force and cursor == date):
            break
        was_fresh = fresh(cursor)
        if not was_fresh:
            validate_day(DB.load('trade_ts', 'day', cursor, vb_level='never'), cursor)
        pending.append((cursor, was_fresh))
        if cursor == boundary:
            break
        previous = int(CALENDAR.td(cursor, -1).as_int())
        if previous >= cursor or previous < boundary:
            raise ValueError(f'Invalid predecessor for adjprice at {cursor}')
        if DB.path('trade_ts', 'adjprice', previous).exists():
            break
        cursor = previous
    changed = False
    for current, was_fresh in reversed(pending):
        # Rebuilding a predecessor invalidates every following chain receipt.
        if was_fresh and not changed and not (force and current == date):
            verified.add(current)
            continue
        validate_day(DB.load('trade_ts', 'day', current, vb_level='never'), current)
        path = DB.path('trade_ts', 'adjprice', current)
        inputs = dependencies(current)
        candidate = calc_adjprice(current)
        old = DB.load('trade_ts', 'adjprice', current, vb_level='never')
        same = (path.exists() and set(old.columns) == set(candidate.columns)
                and old.sort_values('secid').reset_index(drop=True).equals(
                    candidate.reindex(columns=old.columns).sort_values('secid').reset_index(drop=True)))
        action = 'unchanged' if same else 'replace' if path.exists() else 'create'
        if not same:
            atomic_frame(candidate, path)
            DBPath('trade_ts', 'adjprice').clear_date_cache()
        receipt = path.with_suffix('.source.json')
        temp = receipt.with_name(f'.{receipt.name}.{uuid4().hex}')
        try:
            temp.write_text(json.dumps({'inputs': inputs, 'output': stamp(path)}))
            os.replace(temp, receipt)
        finally:
            temp.unlink(missing_ok=True)
        verified.add(current)
        changed = changed or not same
        if actions is not None:
            actions.append((current, action))
    return changed


def ensure_adjprice(date: int, *, force: bool = False, reason: str = 'query') -> bool:
    """Use the previous file as a boundary; backtrack only across missing files."""
    if snapshot_active() and not force:
        # Frozen reads must never certify or replace live files with old inputs.
        day = DB.load('trade_ts', 'day', int(date), vb_level='never')
        adj = DB.load('trade_ts', 'adjprice', int(date), vb_level='never')
        validate_day(day, int(date))
        if adj is None or adj.empty or not covers_day(adj, day, int(date)):
            raise ValueError(f'Incomplete frozen adjprice at {date}; repair inputs before creating a snapshot')
        return False
    with price_lock():
        actions: list = []
        changed = _ensure_adjprice(int(date), force, reason, actions=actions)
        log_summary(reason, actions)
        return changed


def ensure_adjprices(dates, *, force: bool = False, reason: str = 'custom') -> bool:
    """Check a batch under one lock; validate each historical prefix only once."""
    changed = False
    with price_lock():
        verified: set[int] = set()
        actions: list = []
        for date in sorted(set(map(int, dates))):
            # Recheck under the writer lock in case another worker filled it.
            if reason == 'custom' and not force and DB.path('trade_ts', 'adjprice', date).exists():
                continue
            changed = _ensure_adjprice(date, force, reason, verified, actions) or changed
        log_summary(reason, actions)
    return changed


def publish_day(day: pd.DataFrame, date: int, *, repair_history: bool = False) -> bool:
    """Publish raw quotes and fill absent prices; historical propagation is opt-in.

    Use ``repair_history=True`` for an explicit historical correction, or run
    ``ensure_adjprices(..., force=True)`` over the affected historical range.
    """
    date = int(date)
    validate_day(day, date)
    with price_lock():
        path = DB.path('trade_ts', 'day', date)
        old = DB.load('trade_ts', 'day', date, vb_level='never')
        changed = not path.exists() or not old.equals(day)
        action = 'replace' if path.exists() else 'create'
        if changed:
            atomic_frame(day, path)
            DBPath('trade_ts', 'day').clear_date_cache()
        Logger.stdout(f'day date={date} trigger=download action={action if changed else "skip"} '
                      f'source={stamp(path)}', vb_level=1)
        verified: set[int] = set()
        actions: list = []
        if repair_history or not DB.path('trade_ts', 'adjprice', date).exists():
            _ensure_adjprice(date, force=repair_history, reason='download', verified=verified, actions=actions)
        # A corrected historical close can propagate through halted holdings.
        if repair_history:
            for later in DB.dates('trade_ts', 'adjprice'):
                if later > date:
                    _ensure_adjprice(int(later), reason='source-revision', verified=verified, actions=actions)
        log_summary('history-repair' if repair_history else 'download', actions)
        return changed
