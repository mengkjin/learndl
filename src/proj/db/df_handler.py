"""Mapper and processor for dataframes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


from typing import Any , Callable , Iterable , TypeVar

from src.proj.log import Logger
from .code_mapper import secid2secid

__all__ = ['dfHandler']

T = TypeVar('T' , bound = pd.DataFrame | pl.DataFrame)

class dfHandler:
    @classmethod
    def reset_index_pandas(cls , df : pd.DataFrame | Any , reset = True) -> pd.DataFrame:
        """reset index which are not None"""
        if not reset or df is None or df.empty:
            return df
        old_index = [index for index in df.index.names if index]
        df = df.reset_index(old_index , drop = False)
        if isinstance(df.index , pd.RangeIndex):
            df = df.reset_index(drop = True)
        return df

    @classmethod
    def default_mapper(cls , df : T) -> T:
        """reset index which default mapper not None"""
        ret : Any = df
        if isinstance(df , pd.DataFrame):
            if 'date' in df.index.names and 'date' in df.columns:
                df = df.reset_index('date' , drop = True)
            old_index = [idx for idx in df.index.names if idx]
            df = cls.reset_index_pandas(df)
            if 'secid' in df.columns:  
                df['secid'] = secid2secid(df['secid'])
            if old_index: 
                df = df.set_index(old_index)
            ret = df
        elif isinstance(df , pl.DataFrame):
            if 'secid' in df.columns: 
                ret = df.with_columns(pl.from_pandas(secid2secid(df['secid'].to_pandas())).alias('secid')) 
        else:
            raise ValueError(f'Unsupported dataframe type: {type(df)}')
        return ret

    @classmethod
    def load_process_pandas(
        cls , df : pd.DataFrame , date = None, key_column = None , check_na_cols = False , 
        syntax : str = 'some df' , reset_index = True , ignored_fields = [] , indent = 1 , vb_level : Any = 'max'
    ) -> pd.DataFrame:
        """process dataframe , check empty / all-NA and try reset index"""
        if key_column and date is not None: 
            df[key_column] = date

        if df.empty:
            Logger.only_once(f'{syntax} is empty' , mark = f'{syntax} empty' , printer = Logger.alert1 , indent = indent , vb_level = vb_level)
        else:
            na_cols : pd.Series | Any = df.isna().all()
            if na_cols.all():
                Logger.only_once(f'{syntax} is all-NA' , mark = f'{syntax} all_na' , printer = Logger.alert1 , indent = indent , vb_level = vb_level)
            elif check_na_cols and na_cols.any():
                Logger.only_once(f'{syntax} has columns [{str(df.columns[na_cols])}] all-NA' , mark = f'{syntax} all_cols_na' , 
                                 printer = Logger.alert1 , indent = indent , vb_level = vb_level)

        df = cls.reset_index_pandas(df , reset_index)
        if ignored_fields: 
            df = df.drop(columns=ignored_fields , errors='ignore')
        return df

    @classmethod
    def load_process_polars(
        cls , df : pl.DataFrame , date = None, key_column = None , check_na_cols = False , 
        syntax : str = 'some df' , reset_index = True , ignored_fields = [] , indent = 1 , vb_level : Any = 'max'
    ) -> pl.DataFrame:
        """process polars dataframe , check empty / all-NA and try reset index"""
        if key_column and date is not None: 
            if isinstance(date, (pl.Expr, pl.Series, list, tuple, np.ndarray)):
                # if date is a list/array, convert it to pl.Series first
                if not isinstance(date, (pl.Expr, pl.Series)):
                    date = pl.Series(key_column, date)
                df = df.with_columns(date.alias(key_column))
            else:
                df = df.with_columns(pl.lit(date).alias(key_column))

        if len(df) == 0:
            Logger.only_once(f'{syntax} is empty' , mark = f'{syntax} empty' , printer = Logger.alert1 , indent = indent , vb_level = vb_level)
        else:
            def is_all_na_column(col: pl.Series) -> bool:
                if col.dtype in (pl.Float32, pl.Float64):
                    return col.is_null().all() or col.is_nan().all()
                else:
                    return col.is_null().all()
            all_na_flags = [is_all_na_column(df[col]) for col in df.columns]
            all_na = all(all_na_flags)
            if all_na:
                Logger.only_once(f'{syntax} is all-NA' , mark = f'{syntax} all_na' , printer = Logger.alert1 , indent = indent , vb_level = vb_level)
            elif check_na_cols and any(all_na_flags):
                na_cols = [col for col, flag in zip(df.columns, all_na_flags) if flag]
                Logger.only_once(f'{syntax} has columns [{na_cols}] all-NA' , mark = f'{syntax} all_cols_na' , 
                                 printer = Logger.alert1 , indent = indent , vb_level = vb_level)

        if ignored_fields: 
            cols_to_drop = [c for c in ignored_fields if c in df.columns]
            if cols_to_drop:
                df = df.drop(cols_to_drop)
        return df

    @classmethod
    def wrapped_mapper(cls , *mappers : Iterable[Callable[[T], T]] | Callable[[T], T] | None) -> Callable[[T], T]:
        def new_mapper(df : T) -> T:
            return dfHandler.apply_mapper(df , dfHandler.default_mapper , *mappers)
        return new_mapper

    @classmethod
    def apply_mapper(cls , df : T , *mappers : Iterable[Callable[[T], T]] | Callable[[T], T] | None) -> T:
        if (isinstance(df , pd.DataFrame) and df.empty) or (isinstance(df , pl.DataFrame) and len(df) == 0):
            return df

        ret : Any = df
        for mapper in mappers:
            if mapper is None:
                ...
            elif not isinstance(mapper , Iterable):
                ret = mapper(ret)
            else:
                for m in mapper:
                    ret = m(ret)
        return ret