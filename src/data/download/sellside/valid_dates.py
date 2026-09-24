"""Cross-section stats for sellside factors.

File presence answers whether a vendor day was stored. These stats answer
whether that day has a usable cross-section. They live under
``DB_sellside/.data_stats`` so date-folder scans never treat them as factor files.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.proj import DB , Logger
from src.proj.db.interface.db_path import DBPath

STATS_DIR = '.data_stats'
VALID_VALUES_DIR = 'valid_values'
VALUE_COLUMNS = ('date' , 'secid_count' , 'nan_count' , 'is_valid')
_LOCKS: dict[str , threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def stats_root(db_src : str = 'sellside') -> Path:
    """``DB_sellside/.data_stats``, beside the per-key date trees."""
    return DBPath.Parent(db_src).joinpath(STATS_DIR)


def valid_values_path(db_key : str , db_src : str = 'sellside') -> Path:
    """One feather per sellside key, not inside that key's date folders."""
    return stats_root(db_src).joinpath(VALID_VALUES_DIR , f'{db_key}.feather')


def _as_frame(df : pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(df , pd.Series):
        return df.to_frame().T
    return df


def cross_section_stats(df : pd.DataFrame | pd.Series , date : int) -> dict[str , int | bool]:
    """Row count, non-finite factor cells, and whether any finite score remains."""
    frame = _as_frame(df)
    if 'secid' in frame.columns and len(frame):
        secid_count = int(frame['secid'].nunique(dropna = True))
    else:
        secid_count = int(len(frame))
    values = frame.drop(columns = ['secid' , 'date'] , errors = 'ignore')
    if values.empty or values.shape[1] == 0 or len(frame) == 0:
        return {'date' : int(date) , 'secid_count' : secid_count , 'nan_count' : 0 , 'is_valid' : False}
    numeric = values.apply(pd.to_numeric , errors = 'coerce')
    finite = np.isfinite(numeric.to_numpy(dtype = float , na_value = np.nan))
    nan_count = int((~finite).sum())
    return {
        'date' : int(date) ,
        'secid_count' : secid_count ,
        'nan_count' : nan_count ,
        'is_valid' : bool(finite.sum() > 0) ,
    }


def finite_value_count(df : pd.DataFrame | pd.Series) -> int:
    """Count finite factor cells. ``secid`` and ``date`` are identifiers, not scores."""
    frame = _as_frame(df)
    values = frame.drop(columns = ['secid' , 'date'] , errors = 'ignore')
    if values.empty or values.shape[1] == 0 or len(frame) == 0:
        return 0
    numeric = values.apply(pd.to_numeric , errors = 'coerce')
    finite = np.isfinite(numeric.to_numpy(dtype = float , na_value = np.nan))
    return int(finite.sum())


def read_valid_values(path : Path) -> pd.DataFrame | None:
    """Return the stats table, or None when it has not been built."""
    if not path.exists():
        return None
    frame = pd.read_feather(path)
    if frame.empty:
        return pd.DataFrame(columns = list(VALUE_COLUMNS))
    frame['date'] = frame['date'].astype(np.int64)
    frame['secid_count'] = frame['secid_count'].astype(np.int64)
    frame['nan_count'] = frame['nan_count'].astype(np.int64)
    frame['is_valid'] = frame['is_valid'].astype(bool)
    return frame.loc[:, list(VALUE_COLUMNS)].drop_duplicates('date' , keep = 'last').sort_values('date')


def write_valid_values(path : Path , frame : pd.DataFrame) -> None:
    """Replace the stats table atomically."""
    path.parent.mkdir(parents = True , exist_ok = True)
    out = frame.loc[:, list(VALUE_COLUMNS)].copy() if len(frame) else pd.DataFrame(columns = list(VALUE_COLUMNS))
    if len(out):
        out['date'] = out['date'].astype(np.int64)
        out['secid_count'] = out['secid_count'].astype(np.int64)
        out['nan_count'] = out['nan_count'].astype(np.int64)
        out['is_valid'] = out['is_valid'].astype(bool)
        out = out.drop_duplicates('date' , keep = 'last').sort_values('date')
    tmp = path.with_suffix('.feather.tmp')
    out.reset_index(drop = True).to_feather(tmp)
    tmp.replace(path)


def _lock_for(db_src : str , db_key : str) -> threading.Lock:
    name = f'{db_src}/{db_key}'
    with _LOCKS_GUARD:
        lock = _LOCKS.get(name)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[name] = lock
        return lock


def upsert_valid_values(
    db_key : str ,
    rows : pd.DataFrame ,
    db_src : str = 'sellside' ,
) -> pd.DataFrame:
    """Insert or replace stats rows for the given dates."""
    path = valid_values_path(db_key , db_src)
    with _lock_for(db_src , db_key):
        current = read_valid_values(path)
        incoming = rows.loc[:, list(VALUE_COLUMNS)] if len(rows) else pd.DataFrame(columns = list(VALUE_COLUMNS))
        if current is None or current.empty:
            merged = incoming
        elif incoming.empty:
            merged = current
        else:
            merged = pd.concat([current , incoming] , ignore_index = True)
        write_valid_values(path , merged)
        stored = read_valid_values(path)
        return stored if stored is not None else pd.DataFrame(columns = list(VALUE_COLUMNS))


def load_valid_dates(
    db_key : str , db_src : str = 'sellside' , * , required : bool = False ,
) -> np.ndarray | None:
    """Dates with a usable cross-section.

    None means the stats file was never built. ``required`` builds a missing
    file from stored days and then always returns an array. An existing file
    is used as stored; dates absent from it are not scanned.
    """
    path = valid_values_path(db_key , db_src)
    if required and not path.exists():
        Logger.stdout(f'{db_src}.{db_key} valid_values is missing; building it from stored dates')
        backfill_valid_values(db_key , db_src , full = False)
    frame = read_valid_values(path)
    if frame is None:
        return np.array([] , dtype = np.int64) if required else None
    if frame.empty:
        return np.array([] , dtype = np.int64)
    return frame.loc[frame['is_valid'] , 'date'].to_numpy(dtype = np.int64)


def missing_stat_dates(stored : np.ndarray , existing : pd.DataFrame | None) -> np.ndarray:
    """Stored file dates that have no stats row. A missing file means the full history."""
    stored_dates = np.unique(np.asarray(stored , dtype = np.int64))
    if existing is None:
        return stored_dates
    have = existing['date'].to_numpy(dtype = np.int64) if len(existing) else np.array([] , dtype = np.int64)
    return np.setdiff1d(stored_dates , have)


def backfill_valid_values(db_key : str , db_src : str = 'sellside' , * , full : bool = False) -> int:
    """Fill stats for stored days that are absent. ``full`` rebuilds every stored day."""
    stored = DB.dates(db_src , db_key)
    stored_dates = stored.dates if len(stored) else np.array([] , dtype = np.int64)
    path = valid_values_path(db_key , db_src)
    existing = None if full else read_valid_values(path)
    pending = stored_dates if full or existing is None else missing_stat_dates(stored_dates , existing)
    if len(pending) == 0 and existing is not None and not full:
        return 0
    rows = [
        cross_section_stats(DB.load(db_src , db_key , int(day) , vb_level = 'never') , int(day))
        for day in pending
    ]
    frame = pd.DataFrame(rows , columns = list(VALUE_COLUMNS)) if rows else pd.DataFrame(columns = list(VALUE_COLUMNS))
    if full or existing is None:
        write_valid_values(path , frame)
    elif len(frame):
        upsert_valid_values(db_key , frame , db_src)
    return int(len(pending))


def asof_map(requested : np.ndarray , valid : np.ndarray) -> dict[int , int]:
    """Map each request date to the latest valid date on or before it."""
    if len(requested) == 0 or len(valid) == 0:
        return {}
    asked = np.unique(np.asarray(requested , dtype = np.int64))
    known = np.unique(np.asarray(valid , dtype = np.int64))
    pos = np.searchsorted(known , asked , side = 'right') - 1
    usable = pos >= 0
    return {int(day) : int(known[idx]) for day , idx in zip(asked[usable] , pos[usable])}


def stamp_asof(loaded : pd.DataFrame , mapping : Mapping[int , int]) -> pd.DataFrame:
    """Copy each source cross-section onto every request date that as-of maps to it."""
    columns = ['secid' , 'date']
    if loaded.empty or not mapping:
        extra = [c for c in loaded.columns if c not in columns]
        return pd.DataFrame(columns = columns + extra)
    pairs = pd.DataFrame({
        'date' : np.fromiter(mapping.keys() , dtype = np.int64 , count = len(mapping)) ,
        'src' : np.fromiter(mapping.values() , dtype = np.int64 , count = len(mapping)) ,
    })
    base = loaded.rename(columns = {'date' : 'src'})
    out = base.merge(pairs , on = 'src' , how = 'inner').drop(columns = 'src')
    return out.sort_values(['secid' , 'date']).reset_index(drop = True)
