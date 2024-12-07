import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

def get_latest(numerator : str , date : int , **kwargs):
    '''statement@field@fin_type'''
    return DATAVENDOR.get_fin_latest(numerator , date , **kwargs)

def get_hist(numerator : str , date : int , n_last : int = 12 , **kwargs):
    '''statement@field@fin_type'''
    return DATAVENDOR.get_fin_hist(numerator , date , n_last , **kwargs).iloc[:,0]

def hist_zscore(data : pd.Series | pd.DataFrame):
    if isinstance(data , pd.DataFrame):
        print(data)
        assert data.shape[1] == 1 , 'data must be a single column'
        data = data.iloc[:,0]
    grp = data.groupby('secid')
    return (grp.last() - grp.mean()) / grp.std()

def hist_zscore_polars(data : pd.Series):
    df = pl.from_pandas(data.rename('value').to_frame() , include_index=True)
    return df.group_by('secid', maintain_order=True).\
        agg((pl.col('value').last() - pl.col('value').mean()) / pl.col('value').std()).\
            to_pandas().set_index('secid').iloc[:,0]

def get_ratio_latest(numerator : str , denominator : str , date : int , 
                     fin_type : Literal['qtr' , 'ttm' , 'acc'] , numerator_kwargs : dict[str,Any] = {} , 
                     denominator_kwargs : dict[str,Any] = {} , **kwargs):
    '''qtr_method only available in indi'''
    num = get_latest(f'{numerator}@{fin_type}' , date , **numerator_kwargs , **kwargs)
    den = get_latest(f'{denominator}@{fin_type}' , date , **denominator_kwargs , **kwargs)
    return num / den

def get_ratio_zscore(numerator : str , denominator : str , date : int , 
                     numerator_kwargs : dict[str,Any] = {} , 
                     denominator_kwargs : dict[str,Any] = {} , **kwargs):
    '''qtr_method only available in indi'''
    num = get_hist(f'{numerator}@qtr' , date , 12 , **numerator_kwargs , **kwargs)
    den = get_hist(f'{denominator}@qtr' , date , 12 , **denominator_kwargs , **kwargs)
    return hist_zscore(num / den)

class npro_cratio(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '核心净利润占比'
    
    def calc_factor(self, date: int):
        npro = get_latest('is@n_income_attr_p@qtr' , date)
        gp = get_latest('indi@gross_margin@qtr' , date , qtr_method = 'diff')
        biz_tax_surchg = get_latest('is@biz_tax_surchg@qtr' , date)
        sell_exp = get_latest('is@sell_exp@qtr' , date)
        admin_exp = get_latest('is@admin_exp@qtr' , date)
        fin_exp = get_latest('is@fin_exp@qtr' , date)
        rd = get_latest('is@rd_exp@qtr' , date)
        return (gp - biz_tax_surchg - sell_exp - admin_exp - fin_exp - rd) / npro
    
class gp_sales_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM毛利润除以营业收入'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('indi@gross_margin' , 'is@revenue' , date , 'ttm' , numerator_kwargs={'qtr_method':'diff'})

class gp_sales_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度毛利润除以营业收入'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('indi@gross_margin' , 'is@revenue' , date , 'qtr' , numerator_kwargs={'qtr_method':'diff'})

class gp_ta_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM毛利润除以总资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('indi@gross_margin' , 'bs@total_assets' , date , 'ttm' , numerator_kwargs={'qtr_method':'diff'})

class gp_ta_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度毛利润除以总资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('indi@gross_margin' , 'bs@total_assets' , date , 'qtr' , numerator_kwargs={'qtr_method':'diff'})

class gp_ta_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '毛利润/总资产,Z-Score'

    def calc_factor(self, date: int):
        return get_ratio_zscore('indi@gross_margin' , 'bs@total_assets' , date)
    
class npdedt_equ_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM扣非归母净利润/净资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('indi@profit_dedt' , 'bs@total_hldr_eqy_exc_min_int' , date , 'ttm' , numerator_kwargs={'qtr_method':'diff'})

class npdedt_sales_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM扣非归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('indi@profit_dedt' , 'is@revenue' , date , 'ttm' , numerator_kwargs={'qtr_method':'diff'})

class npdedt_ta_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM扣非归母净利润/总资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('indi@profit_dedt' , 'bs@total_assets' , date , 'ttm' , numerator_kwargs={'qtr_method':'diff'})

class npro_equ_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/净资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@n_income_attr_p' , 'bs@total_hldr_eqy_exc_min_int' , date , 'ttm')

class npro_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '归母净利润/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        return get_ratio_zscore('is@n_income_attr_p' , 'bs@total_hldr_eqy_exc_min_int' , date)
    
class npro_equ_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/净资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@n_income_attr_p' , 'bs@total_hldr_eqy_exc_min_int' , date , 'qtr')

class npro_ta_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/总资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@n_income_attr_p' , 'bs@total_assets' , date , 'qtr')

class npro_sales_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@n_income_attr_p' , 'is@revenue' , date , 'ttm')

class npro_sales_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@n_income_attr_p' , 'is@revenue' , date , 'qtr')

class npro_ta_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/总资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@n_income_attr_p' , 'bs@total_assets' , date , 'ttm')

class npro_ta_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/总资产,Z-Score'
    
    def calc_factor(self, date: int):
        return get_ratio_zscore('is@n_income_attr_p' , 'bs@total_assets' , date)

class ocf_sales_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度经营活动现金流/营业收入'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('cf@n_cashflow_act' , 'is@revenue' , date , 'qtr')

class ocf_ta_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度经营活动现金流/总资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('cf@n_cashflow_act' , 'bs@total_assets' , date , 'qtr')

class ocf_sales_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/营业收入'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('cf@n_cashflow_act' , 'is@revenue' , date , 'ttm')

class ocf_ta_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/总资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('cf@n_cashflow_act' , 'bs@total_assets' , date , 'ttm')
    
class op_equ_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM营业利润/净资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@operate_profit' , 'bs@total_hldr_eqy_exc_min_int' , date , 'ttm')

class op_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM营业利润/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        return get_ratio_zscore('is@operate_profit' , 'bs@total_hldr_eqy_exc_min_int' , date)

class op_equ_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度营业利润/净资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@operate_profit' , 'bs@total_hldr_eqy_exc_min_int' , date , 'qtr')

class expense_sales_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度费用/营业收入'
    
    def calc_factor(self, date: int):
        sell_exp = get_latest('is@sell_exp@qtr' , date)
        admin_exp = get_latest('is@admin_exp@qtr' , date)
        fin_exp = get_latest('is@fin_exp@qtr' , date)
        sales = get_latest('is@revenue@qtr' , date)
        return (sell_exp + admin_exp + fin_exp) / sales

class expense_sales_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM费用/营业收入'
    
    def calc_factor(self, date: int):
        sell_exp = get_latest('is@sell_exp@ttm' , date)
        admin_exp = get_latest('is@admin_exp@ttm' , date)
        fin_exp = get_latest('is@fin_exp@ttm' , date)
        sales = get_latest('is@revenue@ttm' , date)
        return (sell_exp + admin_exp + fin_exp) / sales

class roic_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM,EBIT(1-税率)/投入资本'
    
    def calc_factor(self, date: int):
        return get_latest('indi@roic@ttm' , date , qtr_method='diff')

class ebit_ta_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM,EBIT/有形资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@ebit' , 'indi@tangible_asset' , date , 'ttm' , denominator_kwargs={'qtr_method':'exact'})

class sales_ta_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '营业收入/总资产,Z-Score'
    
    def calc_factor(self, date: int):
        return get_ratio_zscore('is@revenue' , 'bs@total_assets' , date)

class tax_equ_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM所得税/净资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@income_tax' , 'bs@total_hldr_eqy_exc_min_int' , date , 'ttm')

class tax_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM所得税/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        return get_ratio_zscore('is@income_tax' , 'bs@total_hldr_eqy_exc_min_int' , date)

class tax_equ_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度所得税/净资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@income_tax' , 'bs@total_hldr_eqy_exc_min_int' , date , 'qtr')

class tp_equ_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM利润总额/净资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@total_profit' , 'bs@total_hldr_eqy_exc_min_int' , date , 'ttm')

class tp_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM利润总额/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        return get_ratio_zscore('is@total_profit' , 'bs@total_hldr_eqy_exc_min_int' , date)

class tp_equ_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度利润总额/净资产'
    
    def calc_factor(self, date: int):
        return get_ratio_latest('is@total_profit' , 'bs@total_hldr_eqy_exc_min_int' , date , 'qtr')
