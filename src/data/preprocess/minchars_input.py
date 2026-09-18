"""Audit min_chars identifiers before building dense historical tensors."""
from __future__ import annotations

import numpy as np
import polars as pl
from pathlib import Path


def validate_keys(df: pl.DataFrame, known: np.ndarray, source: str) -> None:
    if not df.schema['secid'].is_integer() or not df.schema['date'].is_integer():
        raise ValueError(f'{source}: secid/date must be integer keys; got {df.schema}')
    if df['secid'].null_count() or df['date'].null_count():
        raise ValueError(f'{source}: null secid/date keys')
    ids = df['secid'].unique().sort()
    unknown = ids.filter(~ids.is_in(pl.Series(known).implode()))
    if len(unknown):
        raise ValueError(
            f'{source}: {len(ids)} unique secids, {len(unknown)} absent from historical stock metadata '
            f'({len(known)} known). Unknown sample={unknown.head(30).to_list()}. '
            'Stopped before densification. Check the source table/security types and refresh stock metadata; '
            'no rows were silently dropped.'
        )
    if df.select(pl.struct('secid', 'date').is_duplicated().any()).item():
        raise ValueError(f'{source}: duplicate (secid, date) keys; refusing a multiplicative dense join')


def load_selected(path: Path, features: list[str], known: np.ndarray, secid=None) -> pl.DataFrame:
    """Read one file's selected columns, preserving the DB's standard ID mapping."""
    from src.proj.db.basic.df_handler import dfHandler
    columns = ['secid', 'date', *features]
    if path.suffix == '.feather':
        df = pl.read_ipc(path, columns=columns, memory_map=False)
    elif path.suffix == '.parquet':
        df = pl.read_parquet(path, columns=columns)
    else:
        raise ValueError(f'Unsupported min_chars file: {path}')
    df = dfHandler.default_mapper(df)
    if secid is not None:
        df = df.filter(pl.col('secid').is_in(pl.Series(secid).implode()))
    validate_keys(df, known, str(path))
    return df
