"""ABC for database path"""
from __future__ import annotations
from typing import Literal

__all__ = [
    'DF_SUFFIX' , 'SRC_ALTERNATIVES' , 'DB_BY_NAME' , 'DB_BY_DATE' , 
    'EXPORT_BY_NAME' , 'EXPORT_BY_DATE' , 'TAR_SUFFIXES']

DF_SUFFIX   : Literal['feather' , 'parquet'] = 'feather'
TAR_SUFFIXES : tuple[str, ...] = ('.tar' , '.tar.gz' , '.tar.bz2' , '.tar.xz' , '.tar.zst')

SRC_ALTERNATIVES : dict[str , tuple[str,...]] = {
    'trade_ts' : ('trade_js',) ,
    'benchmark_ts' : ('benchmark_js',)
}
DB_BY_NAME  : tuple[str,...] = (
    'information_js' , 'information_ts' , 'index_daily_ts' ,  'index_daily_custom' , 'market_daily'
)
DB_BY_DATE  : tuple[str,...] = (
    'models' , 'sellside' , 'exposure' , 'trade_js' , 'labels_js' , 'benchmark_js' , 
    'trade_ts' , 'financial_ts' , 'analyst_ts' , 'labels_ts' , 'benchmark_ts' , 'membership_ts' , 'holding_ts' ,
    'crawler'
)
EXPORT_BY_NAME : tuple[str,...] = (
    'market_factor' , 'factor_stats_daily' , 'factor_stats_weekly' , 'pooling_weight'
)
EXPORT_BY_DATE : tuple[str,...] = (
    'pred' , 'stock' , 'stock_factor' , 'model_prediction' , 'universe'

)
for name in EXPORT_BY_NAME + EXPORT_BY_DATE:
    assert name not in DB_BY_NAME + DB_BY_DATE , f'{name} must not in DB_BY_NAME and DB_BY_DATE'