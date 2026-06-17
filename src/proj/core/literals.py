"""recurrent Literal types for the project"""
from __future__ import annotations
from typing import Literal , TypeAlias

ALL : TypeAlias = Literal['all']
ANY : TypeAlias = Literal['any']
NONE : TypeAlias = Literal['none']
SELF : TypeAlias = Literal['self']
RANDOM : TypeAlias = Literal['random']
EQUAL : TypeAlias = Literal['equal']

# proj package
VerbosityLevel : TypeAlias = Literal['max','min','never','always'] | int
PandasAccelerator : TypeAlias = Literal['thread' , 'dask' , 'polars' , 'polars_thread']
PolarsAccelerator : TypeAlias = Literal['thread' , 'lazy']

FactorType : TypeAlias = Literal['factor' , 'pred']

FactorMetaType : TypeAlias = Literal['stock' , 'market' , 'affiliate' , 'pooling']
FactorCategory0 : TypeAlias = Literal['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative' , 'market' , 'risk' , 'external' , 'pooling']
FactorCategory1 : TypeAlias = Literal[
    'sellside' ,'weighted' , 'nonlinear' , 'style' , 'market_event' , 'quality' , 'growth' , 'value' , 'earning' , 
    'surprise' , 'coverage' , 'forecast' , 'adjustment' , 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 
    'hf_liquidity' , 'momentum' , 'volatility' , 'correlation' , 'liquidity' , 'holding' , 'trading']

FactorStatsType : TypeAlias = Literal['stats']
FactorStatsPeriod : TypeAlias = Literal['daily' , 'weekly']
FactorFillNanMethod : TypeAlias = Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median']

ICType : TypeAlias = Literal['pearson' , 'spearman']
TradeEngine : TypeAlias = Literal['default' , 'harvest' , 'yale']
TradePrice : TypeAlias = Literal['close' , 'vwap' , 'open']
TradeDirection : TypeAlias = Literal['buy' , 'sell']

intDateType : TypeAlias = Literal['td' , 'cd']
FreqUpdate : TypeAlias = Literal['d' , 'w' , 'm']
FreqFinData : TypeAlias = Literal['y' , 'h' , 'q']
FreqPeriod : TypeAlias = Literal['d' , 'w' , 'm' , 'q' , 'y']
DataUpdateKey : TypeAlias = Literal[
    'crawler_announcement' , 'baostock_5min' , 'rcquant_min' , 'sellside_sql' , 'tushare']

MetricType : TypeAlias = Literal['accuracy' , 'loss' , 'rankic']

DatasetFit : TypeAlias = Literal['train' , 'valid']
DatasetTest : TypeAlias = Literal['test' , 'predict' , 'retrospective']
DatasetAll : TypeAlias = Literal['train' , 'valid' , 'test' , 'predict' , 'retrospective']

StageAll : TypeAlias = Literal['setup' , 'data' , 'fit' , 'test' , 'predict' , 'retrospective' , 'summary']

ConfigWeightScheme : TypeAlias = Literal['equal' , 'top' , 'polar']

ReturnType : TypeAlias = Literal['close' , 'vwap']

DataBlockTimeFrame : TypeAlias = Literal['fit' , 'predict']
DataBlockTimeFrames : TypeAlias = Literal['fit' , 'predict' , 'both']

RunScriptMode : TypeAlias = Literal['shell', 'os']

TimeSeriesAccMethod : TypeAlias = Literal['exp' , 'lin']

_0_3 : TypeAlias = Literal[0,1,2]
_true : TypeAlias = Literal[True]
_false : TypeAlias = Literal[False]