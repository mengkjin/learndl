import numpy as np
import pandas as pd
import statsmodels.api as sm
import polars as pl

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'gp_ta_qoq_trend' , 'gp_ta_yoy_trend' , 'npro_trend' , 'cfo_trend'
]

def calc_trend(data : pd.Series):
    def _trend(args) -> float:
        y = args[0].to_numpy()
        x = np.arange(1, len(y) + 1)
        try:
            return sm.OLS(y, sm.add_constant(x)).fit().params[1] / y.mean()
        except Exception:
            return np.nan
    if not data.name: 
        data = data.rename('data')
    y_name = str(data.name)
    df = pl.from_pandas(data.to_frame() , include_index=True)
    df = df.with_columns(
        pl.when(pl.col(y_name).is_infinite()).then(0).otherwise(pl.col(y_name)).alias(y_name),
    ).drop_nulls()

    df = df.sort(['secid','end_date']).group_by('secid', maintain_order=True).\
        agg(pl.map_groups(exprs=[y_name], function=_trend, return_dtype=pl.Float64 , returns_scalar=True)).\
            to_pandas().set_index('secid').iloc[:,0]
    return df

class gp_ta_qoq_trend(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '单季度毛利润/总资产环比变化趋势'
    
    def calc_factor(self, date: int):
        gp_ta_qoq = DATAVENDOR.get_fin_qoq('gp@qtr / ta@qtr' , date , 20).iloc[:,0]
        return calc_trend(gp_ta_qoq)
    
class gp_ta_yoy_trend(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '单季度毛利润/总资产同比变化趋势'
    
    def calc_factor(self, date: int):
        gp_ta_yoy = DATAVENDOR.get_fin_yoy('gp@qtr / ta@qtr' , date , 20).iloc[:,0]
        return calc_trend(gp_ta_yoy)
    
class npro_trend(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '归母净利润变动趋势'
    
    def calc_factor(self, date: int):
        npro = DATAVENDOR.get_fin_hist('npro@qtr' , date , 20).iloc[:,0]
        return calc_trend(npro)

class cfo_trend(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '经营活动现金流变动趋势'
    
    def calc_factor(self, date: int):
        cfo = DATAVENDOR.get_fin_hist('ncfo@qtr' , date , 20).iloc[:,0]
        return calc_trend(cfo)

