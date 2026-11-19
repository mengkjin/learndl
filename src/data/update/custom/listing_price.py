"""Persist listing reference prices separately from query-time return caches."""
from __future__ import annotations

import os
from uuid import uuid4
import numpy as np
import pandas as pd
from src.proj import DB, Save

__all__ = ['build_listing_prices', 'update_listing_prices']


def build_listing_prices(refresh_date=None, rebuild=False):
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
    available = set(DB.dates('trade_ts', 'day'))
    pending = table.loc[table.adj_list_price.isna() & table.list_dt.isin(available)]
    for date in pending.list_dt.unique():
        day = DB.load('trade_ts', 'day', int(date), vb_level='never')
        if day.empty or not {'secid', 'preclose', 'adjfactor'}.issubset(day.columns):
            continue
        ids = pending.index[pending.list_dt == date]
        day = day.loc[~day.secid.duplicated(keep=False)].set_index('secid').reindex(ids)
        price = pd.to_numeric(day.preclose, errors='coerce')
        factor = pd.to_numeric(day.adjfactor, errors='coerce').fillna(1.)
        valid = np.isfinite(price) & (price > 0) & np.isfinite(factor) & (factor > 0) & np.isfinite(price * factor)
        ids = ids[valid]
        table.loc[ids, 'list_price'] = price.loc[ids]
        table.loc[ids, 'adjfactor'] = factor.loc[ids]
        table.loc[ids, 'adj_list_price'] = (price * factor).loc[ids]
    return table.reset_index()


def update_listing_prices(date):
    table = build_listing_prices(refresh_date=int(date))
    path = DB.path('information_ts', 'list_day_price')
    temporary = path.with_name(f'.{path.stem}.{uuid4().hex}.feather')
    try:
        Save.df(table, temporary, vb_level='never')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
