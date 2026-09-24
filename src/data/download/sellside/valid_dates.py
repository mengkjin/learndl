"""Valid-date index for sellside factors.

File presence answers "has this vendor day been stored". This index answers
"does that day have a usable cross-section". All-null days stay on disk so
download diffs do not fetch them again; as-of reads use only this index.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.proj.db.interface.db_path import DBPath

INDEX_NAME = 'valid_dates.feather'
_LOCKS: dict[str , threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def index_path(db_key : str , db_src : str = 'sellside') -> Path:
    """Sidecar next to the date folders. Its stem is not an 8-digit date."""
    return DBPath(db_src , db_key).parent.joinpath(INDEX_NAME)


def finite_value_count(df : pd.DataFrame | pd.Series) -> int:
    """Count finite factor values. ``secid`` is an identifier, not a score."""
    if isinstance(df , pd.Series):
        df = df.to_frame().T
    if df.empty:
        return 0
    values = df.drop(columns = ['secid' , 'date'] , errors = 'ignore')
    if values.empty or values.shape[1] == 0:
        return 0
    numeric = values.apply(pd.to_numeric , errors = 'coerce')
    return int(np.isfinite(numeric.to_numpy(dtype = float , na_value = np.nan)).sum())


def read_index(path : Path) -> np.ndarray | None:
    """Return sorted dates, or None when the index has not been built."""
    if not path.exists():
        return None
    frame = pd.read_feather(path)
    if frame.empty or 'date' not in frame.columns:
        return np.array([] , dtype = np.int64)
    return np.unique(frame['date'].to_numpy(dtype = np.int64))


def write_index(path : Path , dates : np.ndarray) -> None:
    """Replace the index atomically so a reader never sees a partial file."""
    path.parent.mkdir(parents = True , exist_ok = True)
    tmp = path.with_suffix('.feather.tmp')
    ordered = np.unique(np.asarray(dates , dtype = np.int64))
    pd.DataFrame({'date' : ordered}).to_feather(tmp)
    tmp.replace(path)


def load_valid_dates(db_key : str , db_src : str = 'sellside') -> np.ndarray | None:
    """None means the index was never built; an empty array means no valid day."""
    return read_index(index_path(db_key , db_src))


def _lock_for(db_src : str , db_key : str) -> threading.Lock:
    name = f'{db_src}/{db_key}'
    with _LOCKS_GUARD:
        lock = _LOCKS.get(name)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[name] = lock
        return lock


def apply_validity(
    db_key : str ,
    updates : Mapping[int , bool] ,
    db_src : str = 'sellside' ,
) -> np.ndarray:
    """Insert dates with usable values and drop dates overwritten to all-null.

    A date mapped to False is not inserted. If it was already indexed, it is
    removed so as-of does not keep pointing at an empty file.
    """
    path = index_path(db_key , db_src)
    with _lock_for(db_src , db_key):
        current = read_index(path)
        dates = np.array([] , dtype = np.int64) if current is None else current
        keep = set(int(d) for d in dates)
        for day , is_valid in updates.items():
            day_i = int(day)
            if is_valid:
                keep.add(day_i)
            else:
                keep.discard(day_i)
        ordered = np.array(sorted(keep) , dtype = np.int64)
        write_index(path , ordered)
        return ordered


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
