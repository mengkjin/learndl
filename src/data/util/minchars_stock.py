"""Stock membership for minute characteristics, including delisted stocks."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl


def historical_stock_ids() -> np.ndarray:
    from .stock_info import INFO
    ids = INFO.get_secid()  # No date restriction: retain historical/delisted stocks.
    if len(ids) == 0:
        raise ValueError('Historical stock description is empty; refusing to filter min_chars')
    return ids


def stock_rows(df, known=None, *, source='min_chars'):
    """Filter already DB-mapped IDs; missing stock observations are normal."""
    known = historical_stock_ids() if known is None else known
    if len(known) == 0:
        raise ValueError('Historical stock description is empty; refusing to filter min_chars')
    if isinstance(df, pd.DataFrame):
        if df.empty:
            return df
        ids = df['secid'] if 'secid' in df.columns else df.index.get_level_values('secid')
        result = df.loc[ids.isin(known)]
    else:
        if not df.schema['secid'].is_integer():
            raise ValueError(f'{source}: secid must be integer, got {df.schema["secid"]}')
        result = df.filter(pl.col('secid').is_in(pl.Series(known).implode()))
    if len(result) != len(df):
        logging.getLogger(__name__).warning('%s: removed %d non-stock/null-secid rows; retained %d', source, len(df) - len(result), len(result))
    return result
