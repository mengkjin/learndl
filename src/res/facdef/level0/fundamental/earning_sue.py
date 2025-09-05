import pandas as pd
import numpy as np
import statsmodels.api as sm
import polars as pl
from typing import Literal

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'sue_gp' , 'sue_gp_reg' ,
    'sue_npro' , 'sue_npro_reg' ,
    'sue_op' , 'sue_op_reg' ,
    'sue_tp' , 'sue_tp_reg' ,
    'sue_sales' , 'sue_sales_reg' ,
    'sue_tax' , 'sue_tax_reg'
]

def sue(expression: str , date: int , **kwargs):
    data = DATAVENDOR.get_fin_hist(expression , date , 8 , pivot = False ,**kwargs).iloc[:,0]
    grp = data.groupby('secid')
    return (grp.last() - grp.mean()) / grp.std()

def sue_reg(expression: str , date: int , n_last : int = 8 , **kwargs):
    def _last_resid(args) -> float:
        y = args[0].to_numpy()
        x = np.arange(1, len(y) + 1)
        try:
            return sm.OLS(y, sm.add_constant(x)).fit().resid[-1]
        except Exception as e:
            return np.nan
    
    y_var = DATAVENDOR.get_fin_hist(expression , date , n_last , pivot = False ,**kwargs).iloc[:,0]
    y_name = str(y_var.name)
    df = pl.from_pandas(y_var.to_frame() , include_index=True)
    df = df.with_columns([
        ((pl.col(y_name) - pl.col(y_name).mean().over('secid')) / pl.col(y_name).std().over('secid')).alias(y_name),
    ]).with_columns(
        pl.when(pl.col(y_name).is_infinite()).then(0).otherwise(pl.col(y_name)).alias(y_name),
    ).drop_nulls()

    df = df.sort(['secid','end_date']).group_by('secid', maintain_order=True).\
        agg(pl.map_groups(exprs=[y_name], function=_last_resid, return_dtype=pl.Float64 , returns_scalar=True)).\
            to_pandas().set_index('secid').iloc[:,0]
    return df

class sue_gp(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外毛利润'
    
    def calc_factor(self, date: int):
        return sue('gp@qtr' , date)

class sue_gp_reg(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外毛利润-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('gp@qtr' , date)

class sue_npro(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外归母净利润'
    
    def calc_factor(self, date: int):
        return sue('npro@qtr' , date)

class sue_npro_reg(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外归母净利润-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('npro@qtr' , date)

class sue_op(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业利润'
    
    def calc_factor(self, date: int):
        return sue('oper_np@qtr' , date)

class sue_op_reg(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业利润-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('oper_np@qtr' , date)

class sue_tp(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外利润总额'
    
    def calc_factor(self, date: int):
        return sue('total_np@qtr' , date)

class sue_tp_reg(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外利润总额-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('total_np@qtr' , date)

class sue_sales(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业收入'
    
    def calc_factor(self, date: int):
        return sue('sales@qtr' , date)

class sue_sales_reg(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业收入-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('sales@qtr' , date)

class sue_tax(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外所得税'
    
    def calc_factor(self, date: int):
        return sue('tax@qtr' , date)

class sue_tax_reg(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外所得税-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('tax@qtr' , date)
