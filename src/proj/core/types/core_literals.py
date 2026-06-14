"""recurrent Literal types for the project"""
from __future__ import annotations
from typing import Literal

FactorType = Literal['factor' , 'pred']

FactorMetaType = Literal['stock' , 'market' , 'affiliate' , 'pooling']
FactorCategory0 = Literal['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative' , 'market' , 'risk' , 'external' , 'pooling']
FactorCategory1 = Literal[
    'sellside' ,'weighted' , 'nonlinear' , 'style' , 'market_event' , 'quality' , 'growth' , 'value' , 'earning' , 
    'surprise' , 'coverage' , 'forecast' , 'adjustment' , 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 
    'hf_liquidity' , 'momentum' , 'volatility' , 'correlation' , 'liquidity' , 'holding' , 'trading']

FactorStatsType = Literal['stats']
FactorStatsPeriod = Literal['daily' , 'weekly']
FactorFillNanMethod = Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median']

ICType = Literal['pearson' , 'spearman']

TradeEngine = Literal['default' , 'harvest' , 'yale']
TradePrice = Literal['close' , 'vwap' , 'open']
TradeDirection = Literal['buy' , 'sell']

intDateType = Literal['td' , 'cd']
FreqUpdate = Literal['d' , 'w' , 'm']
FreqFinData = Literal['y' , 'h' , 'q']
FreqPeriod = Literal['d' , 'w' , 'm' , 'q' , 'y']
DataUpdateKey = Literal[
    'crawler_announcement' , 'baostock_5min' , 'rcquant_min' , 'sellside_sql' , 'tushare']

MetricType = Literal['accuracy' , 'loss' , 'rankic']

DatasetFit = Literal['train' , 'valid']
DatasetTest = Literal['test' , 'predict' , 'retrospective']
DatasetAll = Literal['train' , 'valid' , 'test' , 'predict' , 'retrospective']

StageAll = Literal['setup' , 'data' , 'fit' , 'test' , 'predict' , 'retrospective' , 'summary']

ConfigWeightScheme = Literal['equal' , 'top' , 'polar']

ReturnType = Literal['close' , 'vwap']

DataBlockTimeFrame = Literal['fit' , 'predict']
DataBlockTimeFrames = Literal['fit' , 'predict' , 'both']

RunScriptMode = Literal['shell', 'os']

TimeSeriesAccMethod = Literal['exp' , 'lin']

_0_3 = Literal[0,1,2]
_true = Literal[True]
_false = Literal[False]