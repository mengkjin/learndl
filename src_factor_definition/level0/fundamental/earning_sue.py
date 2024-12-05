import pandas as pd
import numpy as np
import statsmodels.api as sm

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR


def last_reg_resid(y : pd.Series):
    x = sm.add_constant(np.arange(1, len(y) + 1))
    model = sm.OLS(y, x).fit()
    return model.resid.iloc[-1]

def sue(numerator: str , date: int , **kwargs):
    data = DATAVENDOR.get_fin_hist(numerator , date , 8 , pivot = False ,**kwargs).iloc[:,0]
    grp = data.groupby('secid')
    return (grp.last() - grp.mean()) / grp.std()

def sue_reg(numerator: str , date: int , **kwargs):
    data = DATAVENDOR.get_fin_hist(numerator , date , 8 , pivot = False ,**kwargs).iloc[:,0]
    grp = data.groupby('secid')
    return grp.apply(last_reg_resid) / grp.std()

class sue_gp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外毛利润'
    
    def calc_factor(self, date: int):
        return sue('indi@qtr@gross_margin' , date , qtr_method = 'diff')

class sue_gp_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外毛利润-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('indi@qtr@gross_margin' , date , qtr_method = 'diff')

class sue_npro(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外归母净利润'
    
    def calc_factor(self, date: int):
        return sue('is@qtr@n_income_attr_p' , date)

class sue_npro_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外归母净利润-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('is@qtr@n_income_attr_p' , date)

class sue_op(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业利润'
    
    def calc_factor(self, date: int):
        return sue('is@qtr@operate_profit' , date)

class sue_op_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业利润-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('is@qtr@operate_profit' , date)

class sue_tp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外利润总额'
    
    def calc_factor(self, date: int):
        return sue('is@qtr@total_profit' , date)

class sue_tp_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外利润总额-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('is@qtr@total_profit' , date)

class sue_sales(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业收入'
    
    def calc_factor(self, date: int):
        return sue('is@qtr@revenue' , date)

class sue_sales_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业收入-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('is@qtr@revenue' , date)

class sue_tax(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外所得税'
    
    def calc_factor(self, date: int):
        return sue('is@qtr@income_tax' , date)

class sue_tax_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外所得税-带截距回归'
    
    def calc_factor(self, date: int):
        return sue_reg('is@qtr@income_tax' , date)
