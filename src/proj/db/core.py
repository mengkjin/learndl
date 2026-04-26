"""ABC for database path"""

import pandas as pd
import polars as pl

from typing import Literal , Union , Iterable , Callable , TypeAlias

__all__ = [
    'DATAFRAME_SUFFIX' , 'SRC_ALTERNATIVES' , 'DB_BY_NAME' , 'DB_BY_DATE' , 
    'EXPORT_BY_NAME' , 'EXPORT_BY_DATE' , 'PL_MAPPER_TYPE' , 'PD_MAPPER_TYPE']

DATAFRAME_SUFFIX   : Literal['feather' , 'parquet'] = 'feather'

SRC_ALTERNATIVES : dict[str , list[str]] = {
    'trade_ts' : ['trade_js'] ,
    'benchmark_ts' : ['benchmark_js']
}
DB_BY_NAME  : list[str] = [
    'information_js' , 'information_ts' , 'index_daily_ts' ,  'index_daily_custom' , 'market_daily']
DB_BY_DATE  : list[str] = [
    'models' , 'sellside' , 'exposure' , 'trade_js' , 'labels_js' , 'benchmark_js' , 
    'trade_ts' , 'financial_ts' , 'analyst_ts' , 'labels_ts' , 'benchmark_ts' , 'membership_ts' , 'holding_ts' ,
    'crawler'
]
EXPORT_BY_NAME : list[str] = ['market_factor' , 'factor_stats_daily' , 'factor_stats_weekly' , 'pooling_weight']
EXPORT_BY_DATE : list[str] = ['pred' , 'stock' , 'stock_factor' , 'model_prediction' , 'universe']
for name in EXPORT_BY_NAME + EXPORT_BY_DATE:
    assert name not in DB_BY_NAME + DB_BY_DATE , f'{name} must not in DB_BY_NAME and DB_BY_DATE'

PL_MAPPER_TYPE : TypeAlias = Union[Iterable[Callable[[pl.DataFrame], pl.DataFrame]] , Callable[[pl.DataFrame], pl.DataFrame] , None]
PD_MAPPER_TYPE : TypeAlias = Union[Iterable[Callable[[pd.DataFrame], pd.DataFrame]] , Callable[[pd.DataFrame], pd.DataFrame] , None]