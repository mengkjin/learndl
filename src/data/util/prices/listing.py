"""Persist listing reference prices separately from query-time return caches."""
from __future__ import annotations

import os
import json
from uuid import uuid4
import numpy as np
import pandas as pd
from src.proj import DB, Logger
from .adjusted import price_lock, atomic_frame, stamp

__all__ = ['build_listing_prices', 'update_listing_prices']


def build_listing_prices(refresh_date=None, rebuild=False, refresh_dates=()):
    desc = DB.load('information_ts', 'description')
    if desc.empty or not {'secid', 'list_dt'}.issubset(desc.columns):
        raise ValueError('Listing prices require valid description metadata')
    table = desc.loc[desc.list_dt > 0, ['secid', 'list_dt']].drop_duplicates('secid').set_index('secid')
    for column in ('list_price', 'adjfactor', 'adj_list_price'):
        table[column] = np.nan
    path = DB.path('information_ts', 'list_day_price')
    if path.exists() and not rebuild:
        old = DB.load('information_ts', 'list_day_price').set_index('secid')
        same = old.reindex(table.index).list_dt.eq(table.list_dt)
        table.loc[same, ['list_price', 'adjfactor', 'adj_list_price']] = old.reindex(table.index).loc[same, ['list_price', 'adjfactor', 'adj_list_price']]
    if refresh_date is not None:
        table.loc[table.list_dt == int(refresh_date), ['list_price', 'adjfactor', 'adj_list_price']] = np.nan
    table.loc[table.list_dt.isin(refresh_dates), ['list_price', 'adjfactor', 'adj_list_price']] = np.nan
    available = set(DB.dates('trade_ts', 'day'))
    if path.exists() and not rebuild:
        lost = old.reindex(table.index).adj_list_price.notna() & table.adj_list_price.isna() & ~table.list_dt.isin(available)
        if lost.any():
            raise ValueError(f'Listing price source missing at {table.loc[lost, "list_dt"].unique().tolist()}')
    pending = table.loc[table.adj_list_price.isna() & table.list_dt.isin(available)]
    for date in pending.list_dt.unique():
        day = DB.load('trade_ts', 'day', int(date), vb_level='never')
        if (day.empty or not {'secid', 'preclose', 'adjfactor'}.issubset(day.columns)
                or day.secid.isna().any() or day.secid.duplicated().any()):
            raise ValueError(f'Invalid listing price source at {date}')
        ids = pending.index[pending.list_dt == date]
        day = day.loc[~day.secid.duplicated(keep=False)].set_index('secid').reindex(ids)
        price = pd.to_numeric(day.preclose, errors='coerce')
        factor = pd.to_numeric(day.adjfactor, errors='coerce').fillna(1.)
        valid = np.isfinite(price) & (price > 0) & np.isfinite(factor) & (factor > 0) & np.isfinite(price * factor)
        if path.exists() and not rebuild:
            existing = old.reindex(ids).adj_list_price.notna()
            if (existing & ~valid).any():
                raise ValueError(f'Invalid replacement listing prices at {date}')
        ids = ids[valid]
        table.loc[ids, 'list_price'] = price.loc[ids]
        table.loc[ids, 'adjfactor'] = factor.loc[ids]
        table.loc[ids, 'adj_list_price'] = (price * factor).loc[ids]
    return table.reset_index()


def update_listing_prices(date=None, *, start=None, end=None, overwrite=False):
    with price_lock():
        path = DB.path('information_ts', 'list_day_price')
        receipt = path.with_suffix('.source.json')
        sources = {str(int(d)): stamp(DB.path('trade_ts', 'day', int(d))) for d in DB.dates('trade_ts', 'day')}
        old = DB.load('information_ts', 'list_day_price') if path.exists() else pd.DataFrame()
        try:
            old_sources = json.loads(receipt.read_text())
            if not isinstance(old_sources, dict):
                raise ValueError('Invalid listing source receipt')
        except (OSError, ValueError):
            old_sources = {}
        changed_dates = [int(d) for d in old_sources.keys() | sources.keys() if old_sources.get(d) != sources.get(d)]
        if not old_sources and not old.empty:
            changed_dates.extend(map(int, old.list_dt.unique()))
        if overwrite:
            changed_dates.extend(int(d) for d in sources if (start is None or int(d) >= start) and (end is None or int(d) <= end))
        table = build_listing_prices(refresh_date=date, refresh_dates=changed_dates)
        changed = not old.equals(table)
        if changed:
            atomic_frame(table, path)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt.with_name(f'.{receipt.name}.{uuid4().hex}')
        try:
            temporary.write_text(json.dumps(sources))
            os.replace(temporary, receipt)
        finally:
            temporary.unlink(missing_ok=True)
        listing_dates = set(table.list_dt)
        relevant = {str(d): sources.get(str(d)) for d in changed_dates if d in listing_dates}
        Logger.stdout(f'listing_price date={date} range={start}..{end} trigger=custom '
                      f'action={"replace" if changed and not old.empty else "create" if changed else "skip"} '
                      f'sources={relevant}', vb_level=1 if changed else 2)
        return changed
