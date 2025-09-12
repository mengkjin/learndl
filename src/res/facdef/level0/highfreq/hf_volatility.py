import polars as pl

from typing import Callable , Literal

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'inday_err_ret' , 'inday_maxdd' , 
    'inday_std_1min' , 'inday_std_5min' , 
    'inday_skewness_1min' , 'inday_skewness_5min' , 
    'inday_kurt_1min' , 'inday_kurt_5min' , 
    'inday_vardown_1min' , 'vardown_intra5min' , 
    'inday_vol_std_1min' , 'inday_vol_std_5min' ,
]

def trailing(date , func : Callable[[int] , pl.DataFrame] , agg : Literal['avg' , 'std' , 'cv' , 'max'] , window : int = 20):
    dates = DATAVENDOR.CALENDAR.td_trailing(date , window)
    df = pl.concat([func(date) for date in dates])
    grp = df.group_by('secid')
    if agg == 'avg':
        df = grp.agg(pl.col('value').mean().alias('value'))
    elif agg == 'std':
        df = grp.agg(pl.col('value').std().alias('value'))
    elif agg == 'cv':
        df = grp.agg((pl.col('value').std() / pl.col('value').mean()).alias('value'))
    elif agg == 'max':
        df = grp.agg(pl.col('value').max().alias('value'))
    else:
        raise ValueError(f'Invalid agg method: {agg}')
    return df.to_pandas().set_index('secid')['value'].sort_index()

class inday_err_ret(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内极端收益'

    def calc_factor(self, date: int):
        def err_ret(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date , with_ret = True)
            df = df.with_columns(
                pl.col('ret').top_k(5).min().over('secid').alias('err_flag') ,
            ).with_columns(
                (pl.col('ret') > pl.col('err_flag')).alias('err_flag')
            ).filter(pl.col('err_flag') == 1).group_by('secid').agg(
                pl.col('ret').mean().alias('value')
            )
            return df
        return trailing(date , err_ret , 'avg' , 20)
    
class inday_maxdd(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内最大回撤'

    def calc_factor(self, date: int):
        def maxdd(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date , with_ret = True)
            df = df.with_columns(
                pl.when(pl.col('minute') == 0).then(pl.col('open')).otherwise(
                    pl.col('high').cum_max().shift(1)).over('secid').alias('cum_high') ,
            ).with_columns(
                (1 - pl.col('low') / pl.col('cum_high')).alias('max_dd')
            ).group_by('secid').agg(
                pl.col('max_dd').max().alias('value')
            )
            return df
        return trailing(date , maxdd , 'max' , 5)
    
class inday_std_1min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内1分钟标准差'

    def calc_factor(self, date: int):
        def std(date : int):
            df = DATAVENDOR.MKLINE.get_1min(date , with_ret = True)
            df = df.group_by('secid').agg(
                pl.col('ret').std().alias('value') ,
            )
            return df
        return trailing(date , std , 'avg' , 20)
    
class inday_std_5min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内5分钟标准差'

    def calc_factor(self, date: int):
        def std(date : int):
            df = DATAVENDOR.MKLINE.get_5min(date , with_ret = True)
            df = df.group_by('secid').agg(
                pl.col('ret').std().alias('value') ,
            )
            return df
        return trailing(date , std , 'avg' , 20)

class inday_skewness_1min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内1分钟偏度'

    def calc_factor(self, date: int):
        def skewness(date : int):
            df = DATAVENDOR.MKLINE.get_1min(date , with_ret = True)
            df = df.group_by('secid').agg(
                pl.col('ret').skew().alias('value') ,
            )
            return df
        return trailing(date , skewness , 'avg' , 20)
    
class inday_skewness_5min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内5分钟偏度'

    def calc_factor(self, date: int):   
        def skewness(date : int):
            df = DATAVENDOR.MKLINE.get_5min(date , with_ret = True)
            df = df.group_by('secid').agg(
                pl.col('ret').skew().alias('value') ,
            )
            return df
        return trailing(date , skewness , 'avg' , 20)
    
class inday_kurt_1min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内1分钟峰度'

    def calc_factor(self, date: int):
        def kurt(date : int):
            df = DATAVENDOR.MKLINE.get_1min(date , with_ret = True)
            df = df.group_by('secid').agg(
                pl.col('ret').kurtosis().alias('value') ,
            )
            return df
        return trailing(date , kurt , 'avg' , 20)
    
class inday_kurt_5min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内5分钟峰度'

    def calc_factor(self, date: int):   
        def kurt(date : int):
            df = DATAVENDOR.MKLINE.get_5min(date , with_ret = True)
            df = df.group_by('secid').agg(
                pl.col('ret').kurtosis().alias('value') ,
            )
            return df
        return trailing(date , kurt , 'avg' , 20)
    
class inday_vardown_1min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内1分钟下行波动占比'

    def calc_factor(self, date: int):
        def vardown(date : int):
            df = DATAVENDOR.MKLINE.get_1min(date , with_ret = True)
            df = df.with_columns(
                pl.when(pl.col('ret') < 0).then(pl.col('ret')).otherwise(None).alias('ret_down')
            ).group_by('secid').agg(
                pl.col('ret').std().alias('std') ,
                pl.col('ret_down').std().alias('std_down') ,
            ).with_columns(
                (pl.col('std_down') / pl.col('std')).alias('value') ,
            )
            return df
        return trailing(date , vardown , 'avg' , 20)
    
class vardown_intra5min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内5分钟下行波动占比'

    def calc_factor(self, date: int):
        def vardown(date : int):
            df = DATAVENDOR.MKLINE.get_5min(date , with_ret = True)
            df = df.with_columns(
                pl.when(pl.col('ret') < 0).then(pl.col('ret')).otherwise(None).alias('ret_down')
            ).group_by('secid').agg(
                pl.col('ret').std().alias('std') ,
                pl.col('ret_down').std().alias('std_down') ,
            ).with_columns(
                (pl.col('std_down') / pl.col('std')).alias('value') ,
            )
            return df
        return trailing(date , vardown , 'avg' , 20)

class inday_vol_std_1min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内1分钟成交量波动率'

    def calc_factor(self, date: int):
        def std(date : int):
            df = DATAVENDOR.MKLINE.get_1min(date)
            df = df.group_by('secid').agg(
                (pl.col('volume').std() / pl.col('volume').mean()).alias('value') ,
            )
            return df
        return trailing(date , std , 'avg' , 20)
    
class inday_vol_std_5min(StockFactorCalculator):
    init_date = 20110101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内5分钟成交量波动率'

    def calc_factor(self, date: int):
        def std(date : int):
            df = DATAVENDOR.MKLINE.get_5min(date)
            df = df.group_by('secid').agg(
                (pl.col('volume').std() / pl.col('volume').mean()).alias('value') ,
            )
            return df
        return trailing(date , std , 'avg' , 20)