import pandas as pd
import numpy as np
import polars as pl
import statsmodels.api as sm

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'lpnp' , 'ocfa' , 'rroc'
]

def ts_last_resid_polars(y_var : str | pd.Series , x_vars : list[str | pd.Series] , date : int , n_last : int = 12):
    def _last_resid(args) -> pl.Series:
        y = args[0].to_numpy()
        if len(args) > 2:
            x = np.stack([arg.to_numpy() for arg in args[1:]] , axis=-1)
        else:
            x = args[1].to_numpy()
        try:
            return pl.Series(sm.OLS(y, sm.add_constant(x)).fit().resid[-1:], dtype=pl.Float64)
        except Exception as e:
            return pl.Series([np.nan], dtype=pl.Float64)
    
    if isinstance(y_var , str): y_var = DATAVENDOR.get_fin_hist(y_var , date , n_last).iloc[:,0]
    assert y_var.name , 'y_var must have a name'
    y_name = str(y_var.name)

    df = pl.from_pandas(y_var.to_frame() , include_index=True)
    for x_var in x_vars:
        if isinstance(x_var , str): x_var = DATAVENDOR.get_fin_hist(x_var , date , n_last).iloc[:,0]
        assert x_var.name , 'x_vars must have a name'
        df = df.join(pl.from_pandas(x_var.to_frame() , include_index=True) , on = ['secid' , 'end_date'])
    cols = [col for col in df.columns if col not in ['secid', 'end_date']]

    df = df.with_columns([
        *[((pl.col(col) - pl.col(col).mean().over('secid')) / pl.col(col).std().over('secid')).alias(col) for col in cols],
    ]).with_columns(
        *[pl.when(pl.col(col).is_infinite()).then(0).otherwise(pl.col(col)).alias(col) for col in cols],
    ).drop_nulls(y_name).fill_null(0).fill_nan(0)

    df = df.sort(['secid','end_date']).group_by('secid', maintain_order=True).agg(
        pl.map_groups(exprs=cols, function=_last_resid)).to_pandas().set_index('secid').iloc[:,0]

    return df

class lpnp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '线性提纯净利润'
    
    def calc_factor(self, date: int):        
        df = ts_last_resid_polars(y_var='npro@qtr' , x_vars=['is@total_revenue@qtr - sales@qtr' , 'bs@cip@qtr'] , date = date)
        return df
    
class ocfa(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '产能利用率提升'
    
    def calc_factor(self, date: int):
        df = ts_last_resid_polars(y_var='is@oper_cost@qtr' , x_vars=['bs@fix_assets@qtr'] , date = date)
        return df
    
class rroc(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '营业能力改善'
    
    def calc_factor(self, date: int):
        df = ts_last_resid_polars(y_var='sales@qtr' , x_vars=['is@oper_cost@qtr'] , date = date)
        return df