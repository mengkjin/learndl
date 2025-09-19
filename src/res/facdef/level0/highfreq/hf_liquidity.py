import pandas as pd
import polars as pl

from typing import Literal , Callable

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'inday_smart_money' , 'inday_stupid_money' , 
    'inday_vol_utd' , 'inday_vol_coefvar' , 
    'inday_vol_end15min' , 'inday_vol_st5min' , 
    'inday_volpct_phigh' , 'inday_volpct_plow' , 
    'inday_volpct_devhigh' , 
    'vol_high_std' ,
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

class inday_smart_money(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_liquidity'
    description = '聪明钱因子'

    def calc_factor(self, date: int):
        def inday_pct(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            day_vol = df.group_by('secid').agg(pl.sum('volume').alias('day_value'))
            seg_vol = df.with_columns(
                pl.col('close').pct_change().shift(-1).rank(method = 'ordinal').over('secid').alias('next_ret')
            ).with_columns(
                (pl.col('next_ret') / pl.col('next_ret').max() >= 0.9).over('secid').alias('smart_money')
            ).filter(pl.col('smart_money') == 1).group_by('secid').agg(pl.sum('volume').alias('seg_value'))
            df = day_vol.join(seg_vol , on = ['secid'] , how = 'left').with_columns(
                (pl.col('seg_value') / pl.col('day_value')).alias('value'))
            return df.select(['secid' , 'value'])
        return trailing(date , inday_pct , 'avg' , 20)
    
class inday_stupid_money(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_liquidity'
    description = '傻钱因子'

    def calc_factor(self, date: int):
        def inday_pct(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            day_vol = df.group_by('secid').agg(pl.sum('volume').alias('day_value'))
            seg_vol = df.with_columns(
                pl.col('close').pct_change().shift(-1).rank(method = 'ordinal').over('secid').alias('next_ret')
            ).with_columns(
                (pl.col('next_ret') / pl.col('next_ret').max() <= 0.1).over('secid').alias('stupid_money')
            ).filter(pl.col('stupid_money') == 1).group_by('secid').agg(pl.sum('volume').alias('seg_value'))
            df = day_vol.join(seg_vol , on = ['secid'] , how = 'left').with_columns(
                (pl.col('seg_value') / pl.col('day_value')).alias('value'))
            return df.select(['secid' , 'value'])
        return trailing(date , inday_pct , 'avg' , 20)
    
class inday_vol_utd(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_liquidity'
    description = '20日换手率分布因子'

    def calc_factor(self, date: int):
        def inday_std(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            df = df.group_by('secid').agg(pl.std('volume').alias('value'))
            return df
        return trailing(date , inday_std , 'cv' , 20)
    
class inday_vol_coefvar(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_liquidity'
    description = '20日变异数比率因子'

    def calc_factor(self, date: int):
        def inday_cv(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            df = df.group_by('secid').agg((pl.std('volume') / pl.mean('volume')).alias('value'))
            return df
        return trailing(date , inday_cv , 'avg' , 20)
    
class inday_vol_end15min(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_liquidity'
    description = '尾盘成交占比'

    def calc_factor(self, date: int):
        def inday_pct(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            min_thres = 225. if max(df['minute']) > 200. else 45.
            day_vol = df.group_by('secid').agg(pl.sum('volume').alias('day_value'))
            seg_vol = df.filter(pl.col('minute') >= min_thres).group_by('secid').agg(pl.sum('volume').alias('seg_value'))
            df = day_vol.join(seg_vol , on = ['secid'] , how = 'left').with_columns(
                (pl.col('seg_value') / pl.col('day_value')).alias('value'))
            return df.select(['secid' , 'value'])
        return trailing(date , inday_pct , 'avg' , 20)
    
class inday_vol_st5min(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_liquidity'
    description = '开盘成交占比'

    def calc_factor(self, date: int):
        def inday_pct(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            min_thres = 5. if max(df['minute']) > 200 else 1.
            day_vol = df.group_by('secid').agg(pl.sum('volume').alias('day_value'))
            seg_vol = df.filter(pl.col('minute') < min_thres).group_by('secid').agg(pl.sum('volume').alias('seg_value'))
            df = day_vol.join(seg_vol , on = ['secid'] , how = 'left').with_columns(
                (pl.col('seg_value') / pl.col('day_value')).alias('value'))
            return df.select(['secid' , 'value'])
        return trailing(date , inday_pct , 'avg' , 20)
    
class inday_volpct_phigh(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_liquidity'
    description = '高价格成交占比'

    def calc_factor(self, date: int):
        def inday_pct(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            day_vol = df.group_by('secid').agg(pl.sum('volume').alias('day_value'))
            seg_vol = df.with_columns(
                (pl.col('volume').rank(method = 'ordinal') / pl.col('volume').count() >= 0.9).over('secid').alias('flag')
            ).filter(pl.col('flag') == 1).group_by('secid').agg(pl.sum('volume').alias('seg_value'))
            df = day_vol.join(seg_vol , on = ['secid'] , how = 'left').with_columns(
                (pl.col('seg_value') / pl.col('day_value')).alias('value'))
            return df.select(['secid' , 'value'])
        return trailing(date , inday_pct , 'avg' , 20)
    
class inday_volpct_plow(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_liquidity'
    description = '低价格成交占比'

    def calc_factor(self, date: int):
        def inday_pct(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            day_vol = df.group_by('secid').agg(pl.sum('volume').alias('day_value'))
            seg_vol = df.with_columns(
                (pl.col('volume').rank(method = 'ordinal') / pl.col('volume').count() <= 0.1).over('secid').alias('flag')
            ).filter(pl.col('flag') == 1).group_by('secid').agg(pl.sum('volume').alias('seg_value'))
            df = day_vol.join(seg_vol , on = ['secid'] , how = 'left').with_columns(
                (pl.col('seg_value') / pl.col('day_value')).alias('value'))
            return df.select(['secid' , 'value'])
        return trailing(date , inday_pct , 'avg' , 20)
    
class inday_volpct_devhigh(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_liquidity'
    description = '高价格波动成交占比'

    def calc_factor(self, date: int):
        def inday_pct(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            day_vol = df.group_by('secid').agg(pl.sum('volume').alias('day_value'))
            seg_vol = df.with_columns(
                pl.col('close').diff().abs().over('secid').alias('vol_dev')
            ).with_columns(
                (pl.col('vol_dev') / pl.col('vol_dev').max() >= 0.9).over('secid').alias('flag')
            ).filter(pl.col('flag') == 1).group_by('secid').agg(pl.sum('volume').alias('seg_value'))
            df = day_vol.join(seg_vol , on = ['secid'] , how = 'left').with_columns(
                (pl.col('seg_value') / pl.col('day_value')).alias('value'))
            return df.select(['secid' , 'value'])
        return trailing(date , inday_pct , 'avg' , 20)
    
class vol_high_std(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '高波动交易日成交量占比'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)
        def ret_std(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date ,  with_ret = True)
            df = df.group_by('secid').agg(
                pl.col('ret').std().alias('std') ,
                pl.lit(date).alias('date')
            )
            return df.select('secid' , 'date' , 'std')
        df = pl.concat([ret_std(date) for date in dates])
        df = df.with_columns(
            pl.col('std').rank().over('secid').alias('top_std')
        ).with_columns((pl.col('top_std') > 15.0).alias('flag'))
        df = df.to_pandas().set_index('secid')
        day_vols = DATAVENDOR.TRADE.get_volumes(min(dates) , date , pivot = False , mask = False)
        df = df.merge(day_vols , on = ['secid' , 'date'] , how = 'inner')
        
        hvol_v = df.query('flag == 1').groupby('secid')['volume'].sum()
        all_v = df.groupby('secid')['volume'].sum()
        assert isinstance(hvol_v , pd.Series) , f'hvol_v must be a pandas series, but got {type(hvol_v)}'
        assert isinstance(all_v , pd.Series) , f'all_v must be a pandas series, but got {type(all_v)}'
        return hvol_v / all_v
