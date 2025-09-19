import pandas as pd
import polars as pl

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator

from src.func.transform import neutral_resid

__all__ = [
    'inday_amap_orig' , 'inday_conf_persist' , 'inday_regain_conf_persist' , 'inday_high_time' ,
    'inday_incvol_mom' , 'inday_trend_avg' , 'inday_trend_std' ,
    'inday_vwap_diff_hlvol' ,
    'mom_high_volcv' ,  'mom_high_pstd' ,
]

def neutral_resid_pl(df : pl.DataFrame , x : str , y : str , over = 'secid'):
    df = df.with_columns(
        (pl.corr(x , y) * pl.col(y).std() / pl.col(x).std()).over(over).alias('beta')
    ).with_columns(
        (pl.col(y) - pl.col('beta') * pl.col(x)).alias('resid')
    ).with_columns(
        pl.col('resid').mean().over(over).alias('intercept')
    ).with_columns(
        (pl.col('resid') - pl.col('intercept')).alias('resid')
    )
    return df

class inday_amap_orig(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = 'APM原始值,上下午价格行为差异'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)
        mom20 = DATAVENDOR.TRADE.get_returns(min(dates) , date , return_type = 'close' , pivot = False , mask = False)
        mom20 = (mom20 + 1).groupby('secid')['pctchange'].prod() - 1

        def ampm(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date , with_ret = True)
            min_thres = 120. if max(df['minute']) > 200. else 24.
            df = df.with_columns(
                (pl.when(pl.col('minute') < min_thres).then(0).otherwise(1)).alias('time_flag')
            )
            mkt = df.group_by(['time_flag' , 'minute']).agg(
                pl.col('ret').mean().alias('mkt')
            ).group_by('time_flag').agg(
                ((pl.col('mkt') + 1).product() - 1).alias('mkt')
            )

            df = df.group_by(['time_flag' , 'secid']).agg(
                ((pl.col('ret') + 1).product() - 1).alias('ret')
            ).with_columns(
                pl.lit(date).alias('date')
            ).join(mkt , on = ['time_flag'] , how = 'inner')
            return df

        apm = pl.concat([ampm(date) for date in dates])
        apm = neutral_resid_pl(apm , 'ret' , 'mkt' , 'secid')
        apm = apm.filter(pl.col('time_flag') == 0).join(
            apm.filter(pl.col('time_flag') == 1) , on = ['secid' , 'date'] , how = 'inner'
        ).with_columns(
            (pl.col('resid') / pl.col('resid_right')).alias('delta')
        ).group_by('secid').agg(
            (pl.col('delta').mean() / pl.col('delta').std() / pl.col('delta').count().sqrt()).alias('apm') 
        ).to_pandas().set_index('secid')['apm'].reindex(mom20.index)
        apm = neutral_resid(apm , mom20)
        return apm

    
class inday_conf_persist(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '过度自信因子'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)

        def conf_persist(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date , with_ret = True)
            ret = df.group_by('secid').agg(((pl.col('ret') + 1).product() - 1).alias('ret'))
            df = df.with_columns(
                pl.col('ret').mean().over('secid').alias('ret_mean') , pl.col('ret').std().over('secid').alias('ret_std') ,
            ).with_columns(
                (pl.col('ret_mean') + pl.col('ret_std')).alias('upper') , (pl.col('ret_mean') - pl.col('ret_std')).alias('lower')
            ).with_columns(
                (pl.col('ret') > pl.col('upper')).alias('up') , (pl.col('ret') < pl.col('lower')).alias('down')
            )
            cp_down = df.filter(pl.col('down') == 1).group_by('secid').agg(pl.col('minute').median().alias('cp_down'))
            cp_up = df.filter(pl.col('up') == 1).group_by('secid').agg(pl.col('minute').median().alias('cp_up'))
            df = cp_down.join(cp_up , on = 'secid' , how = 'inner').with_columns(
                (pl.col('cp_down') - pl.col('cp_up')).alias('conf_persist')
            ).with_columns(
                pl.lit(date).alias('date')
            ).join(ret , on = 'secid' , how = 'inner')
            return df.select('secid' , 'date' , 'ret' , 'conf_persist')

        cp = pl.concat([conf_persist(date) for date in dates])
        cp = cp.group_by('secid').agg(
            pl.col('conf_persist').mean().alias('cp_mean') ,pl.col('conf_persist').std().alias('cp_std') ,
        ).with_columns(
            (pl.col('cp_mean').rank() - pl.col('cp_std').rank()).alias('conf_persist')
        )
        cp = cp.to_pandas().set_index('secid')['conf_persist']
        return cp
    
class inday_regain_conf_persist(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '重拾自信因子'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)

        def conf_persist(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date , with_ret = True)
            ret = df.group_by('secid').agg(((pl.col('ret') + 1).product() - 1).alias('ret'))
            df = df.with_columns(
                pl.col('ret').mean().over('secid').alias('ret_mean') , pl.col('ret').std().over('secid').alias('ret_std') ,
            ).with_columns(
                (pl.col('ret_mean') + pl.col('ret_std')).alias('upper') , (pl.col('ret_mean') - pl.col('ret_std')).alias('lower')
            ).with_columns(
                (pl.col('ret') > pl.col('upper')).alias('up') , (pl.col('ret') < pl.col('lower')).alias('down')
            )
            cp_down = df.filter(pl.col('down') == 1).group_by('secid').agg(pl.col('minute').median().alias('cp_down'))
            cp_up = df.filter(pl.col('up') == 1).group_by('secid').agg(pl.col('minute').median().alias('cp_up'))
            df = cp_down.join(cp_up , on = 'secid' , how = 'inner').with_columns(
                (pl.col('cp_down') - pl.col('cp_up')).alias('conf_persist')
            ).with_columns(
                pl.lit(date).alias('date')
            ).join(ret , on = 'secid' , how = 'inner')
            return df.select('secid' , 'date' , 'ret' , 'conf_persist')
        
        rcp = pl.concat([conf_persist(date) for date in dates])
        rcp = neutral_resid_pl(rcp , 'ret' , 'conf_persist' , 'date')
        rcp = rcp.group_by('secid').agg(
            pl.col('resid').mean().alias('rcp_mean') ,
            pl.col('resid').std().alias('rcp_std') ,
        ).with_columns(
            (pl.col('rcp_mean').rank() - pl.col('rcp_std').rank()).alias('regain_conf_persist')
        )
        rcp = rcp.to_pandas().set_index('secid')['regain_conf_persist']
        return rcp
    
class inday_high_time(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '日内高点位置'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)
        def high_time(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date , with_ret = True)
            df = df.with_columns(
                pl.col('high').top_k(5).min().over('secid').alias('high_flag') ,
            ).with_columns(
                (pl.col('high') >= pl.col('high_flag')).alias('high_flag')
            ).filter(pl.col('high_flag') == 1).group_by('secid').agg(
                pl.col('minute').median().alias('high_flag')
            )
            return df.select('secid' , 'high_flag')
        
        df = pl.concat([high_time(date) for date in dates])
        df = df.group_by('secid').agg(pl.col('high_flag').mean().alias('high_flag'))
        df = df.to_pandas().set_index('secid')['high_flag']
        return df
    
class inday_incvol_mom(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '量升累计收益'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)
        def incvol_ret(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date , with_ret = True)
            df = df.with_columns(
                (pl.col('volume') > pl.col('volume').shift(1)).over('secid').alias('incvol_flag') 
            ).filter(pl.col('incvol_flag') == 1).group_by('secid').agg(
                pl.col('ret').sum().alias('incvol_ret')
            )
            return df.select('secid' , 'incvol_ret')
        df = pl.concat([incvol_ret(date) for date in dates])
        df = df.group_by('secid').agg(
            pl.col('incvol_ret').sum().alias('incvol_ret')
        ).to_pandas().set_index('secid')['incvol_ret']
        return df
    
class inday_trend_avg(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '日内价格变化路径'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)
        mom20 = DATAVENDOR.TRADE.get_returns(min(dates) , date , return_type = 'close' , pivot = False , mask = False)
        mom20 = (mom20 + 1).groupby('secid')['pctchange'].prod() - 1
        def price_trend(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            df = df.group_by('secid').agg(
                (pl.corr('vwap' , 'minute') * pl.col('vwap').std() / pl.col('minute').std()).alias('trend')
            )
            return df.select('secid' , 'trend')

        trend = pl.concat([price_trend(date) for date in dates]).group_by('secid').agg(
            pl.col('trend').mean().alias('trend_avg')
        ).to_pandas().set_index('secid')['trend_avg'].reindex(mom20.index)
        trend = neutral_resid(trend , mom20)
        return trend
    
class inday_trend_std(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '日内价格变化路径'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)
        mom20 = DATAVENDOR.TRADE.get_returns(min(dates) , date , return_type = 'close' , pivot = False , mask = False)
        mom20 = mom20.groupby('secid')['pctchange'].std()
        assert isinstance(mom20 , pd.Series) , f'mom20 must be a pandas series, but got {type(mom20)}'

        def price_trend(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            df = df.group_by('secid').agg(
                (pl.corr('vwap' , 'minute') * pl.col('vwap').std() / pl.col('minute').std()).alias('trend')
            )
            return df.select('secid' , 'trend')
        
        trend = pl.concat([price_trend(date) for date in dates]).group_by('secid').agg(
            pl.col('trend').std().alias('trend_std')
        ).to_pandas().set_index('secid')['trend_std'].reindex(mom20.index)
        trend = neutral_resid(trend , mom20)
        return trend
    

    
class inday_vwap_diff_hlvol(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '日内高低成交量vwap差'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)
        def high_vol(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date , with_ret = True)
            df = df.with_columns(
                pl.col('volume').median().over('secid').alias('vol_flag')
            ).with_columns(
                (pl.col('volume') >= pl.col('vol_flag')).alias('vol_flag')
            )
            return df.select('secid' , 'vol_flag' , 'vwap')
        df = pl.concat([high_vol(date) for date in dates])
        vwap_h = df.filter(pl.col('vol_flag') == 1).group_by('secid').agg(
            pl.col('vwap').mean().alias('vwap_h')
        )
        vwap_l = df.filter(pl.col('vol_flag') == 0).group_by('secid').agg(
            pl.col('vwap').mean().alias('vwap_l')
        )
        df = vwap_h.join(vwap_l , on = 'secid' , how = 'inner').with_columns(
            (pl.col('vwap_h') - pl.col('vwap_l')).alias('vwap_diff') ,
            ((pl.col('vwap_h') + pl.col('vwap_l')) / 2).alias('vwap_avg') ,
        ).with_columns(
            (pl.col('vwap_diff') / pl.col('vwap_avg')).alias('vwap_diff_pct')
        )
        return df.to_pandas().set_index('secid')['vwap_diff_pct']
    
class mom_high_volcv(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '分钟成交量波动最大区间的动量因子'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)
        def vol_z(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date)
            df = df.group_by('secid').agg(
                (pl.col('volume').std() / pl.col('volume').mean()).alias('vol_z') ,
                pl.lit(date).alias('date')
            )
            return df.select('secid' , 'date' , 'vol_z')
        df = pl.concat([vol_z(date) for date in dates])
        df = df.with_columns(
            pl.col('vol_z').rank().over('secid').alias('top_z')
        ).with_columns(
            (pl.col('top_z') > 15.0).alias('top_z_flag')
        )
        df = df.filter(pl.col('top_z_flag') == 1).to_pandas().set_index('secid')
        day_rets = DATAVENDOR.TRADE.get_returns(min(dates) , date , return_type = 'close' , pivot = False , mask = False)
        df = df.merge(day_rets , on = ['secid' , 'date'] , how = 'inner')
        return df.groupby('secid')['pctchange'].mean()
    
class mom_high_pstd(StockFactorCalculator):
    init_date = 20110101
    category1 = 'hf_momentum'
    description = '日内高波动累计动量'

    def calc_factor(self, date: int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 20)
        def vol_z(date : int):
            df = DATAVENDOR.MKLINE.get_kline(date , with_ret = True)
            df = df.group_by('secid').agg(
                (pl.col('ret').std()).alias('ret_std') ,
                pl.lit(date).alias('date')
            )
            return df.select('secid' , 'date' , 'ret_std')
        df = pl.concat([vol_z(date) for date in dates])
        df = df.with_columns(
            pl.col('ret_std').rank().over('secid').alias('top_std')
        ).with_columns(
            (pl.col('top_std') > 15.0).alias('top_std_flag')
        )
        df = df.filter(pl.col('top_std_flag') == 1).to_pandas().set_index('secid')
        day_rets = DATAVENDOR.TRADE.get_returns(min(dates) , date , return_type = 'close' , pivot = False , mask = False)
        df = df.merge(day_rets , on = ['secid' , 'date'] , how = 'inner')
        return df.groupby('secid')['pctchange'].mean()