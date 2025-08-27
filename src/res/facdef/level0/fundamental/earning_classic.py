import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'npro_core_ratio' , 'gp_margin_qtr' , 'gp_margin_ttm' , 'gp_margin_zscore' ,
    'gp_ta_qtr' , 'gp_ta_ttm' , 'gp_ta_zscore' ,
    'dedt_equ_ttm' , 'dedt_sales_ttm' , 'dedt_ta_ttm' ,
    'roe_qtr' , 'roe_ttm' , 'roe_zscore' ,
    'roa_qtr' , 'roa_ttm' , 'roa_zscore' ,
    'net_margin_qtr' , 'net_margin_ttm' , 'net_margin_zscore' ,
    'cfo_sales_qtr' , 'cfo_sales_ttm' , 'cfo_sales_zscore' ,
    'cfo_ta_qtr' , 'cfo_ta_ttm' , 'cfo_ta_zscore' ,
    'oper_margin_qtr' , 'oper_margin_ttm' , 'oper_margin_zscore' ,
    'expense_sales_qtr' , 'expense_sales_ttm' , 'expense_sales_zscore' ,
    'roic_qtr' , 'roic_ttm' , 'roic_zscore' ,
    'ebit_tangible_ttm' , 
    'tax_equ_qtr' , 'tax_equ_ttm' , 'tax_equ_zscore'
]

def get_hist_zscore(expression : str , date : int):
    data = DATAVENDOR.get_fin_hist(expression , date , 12).iloc[:,0]
    grp = data.groupby('secid')
    return (grp.last() - grp.mean()) / grp.std()

def get_hist_zscore_polars(expression : str , date : int):
    data = DATAVENDOR.get_fin_hist(expression , date , 12).iloc[:,0]
    df = pl.from_pandas(data.rename('value').to_frame() , include_index=True)
    return df.group_by('secid', maintain_order=True).\
        agg((pl.col('value').last() - pl.col('value').mean()) / pl.col('value').std()).\
            to_pandas().set_index('secid').iloc[:,0]

class npro_core_ratio(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '核心净利润占比'
    
    def calc_factor(self, date: int):
        npro = DATAVENDOR.get_fin_latest('npro@ttm' , date)
        gp = DATAVENDOR.get_fin_latest('gp@ttm' , date)
        subtracts = [DATAVENDOR.get_fin_latest(expr , date).reindex(gp.index).fillna(0) 
                    for expr in ['is@biz_tax_surchg@ttm' , 'is@sell_exp@ttm' , 'is@admin_exp@ttm' , 'is@fin_exp@ttm' , 'is@rd_exp@ttm']]
        core = gp - sum(subtracts)
        return core / npro
    
class gp_margin_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度毛利润除以营业收入(毛利率)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('gp@qtr / sales@qtr' , date)
class gp_margin_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM毛利润除以营业收入(毛利率)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('gp@ttm / sales@ttm' , date)
    
class gp_margin_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM毛利润除以营业收入(毛利率),Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('gp@ttm / sales@ttm' , date)

class gp_ta_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度毛利润除以总资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('gp@qtr / ta@qtr' , date)

class gp_ta_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM毛利润除以总资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('gp@ttm / ta@ttm' , date)

class gp_ta_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '毛利润/总资产,Z-Score'

    def calc_factor(self, date: int):
        return get_hist_zscore('gp@qtr / ta@qtr' , date)
    
class dedt_equ_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM扣非归母净利润/净资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('dedt@ttm / equ@ttm' , date)

class dedt_sales_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM扣非归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('dedt@ttm / sales@ttm' , date)

class dedt_ta_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM扣非归母净利润/总资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('dedt@ttm / ta@ttm' , date)

class roe_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/净资产(ROE)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('npro@qtr / equ@qtr' , date)
class roe_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/净资产(ROE)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('npro@ttm / equ@ttm' , date)

class roe_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '归母净利润/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('npro@qtr / equ@qtr' , date)
    

class roa_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/总资产(ROA)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('npro@qtr / ta@qtr' , date)
    
class roa_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/总资产(ROA)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('npro@ttm / ta@ttm' , date)

class roa_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/总资产(ROA),Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('npro@qtr / ta@qtr' , date)

class net_margin_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/营业收入(净利率)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('npro@qtr / sales@qtr' , date)
class net_margin_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/营业收入(净利率)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('npro@ttm / sales@ttm' , date)
    
class net_margin_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/营业收入(净利率),Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('npro@qtr / sales@qtr' , date)

class cfo_sales_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度经营活动现金流/营业收入'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('ncfo@qtr / sales@qtr' , date)
    
class cfo_sales_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/营业收入'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('ncfo@ttm / sales@ttm' , date)

class cfo_sales_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/营业收入,Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('ncfo@qtr / sales@qtr' , date)

class cfo_ta_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度经营活动现金流/总资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('ncfo@qtr / ta@qtr' , date)

class cfo_ta_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/总资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('ncfo@ttm / ta@ttm' , date)
    
class cfo_ta_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/总资产,Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('ncfo@qtr / ta@qtr' , date)
    
class oper_margin_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度营业利润/营业收入(营业利润率)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('oper_np@qtr / sales@qtr' , date)
    
class oper_margin_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM营业利润/营业收入(营业利润率)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('oper_np@ttm / sales@ttm' , date)

class oper_margin_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM营业利润/营业收入(营业利润率),Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('oper_np@qtr / sales@qtr' , date)

class expense_sales_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度费用/营业收入'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('(is@sell_exp@qtr + is@admin_exp@qtr + is@fin_exp@qtr) / sales@qtr' , date)

class expense_sales_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM费用/营业收入'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('(is@sell_exp@ttm + is@admin_exp@ttm + is@fin_exp@ttm) / sales@ttm' , date)
    
class expense_sales_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM费用/营业收入,Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('(is@sell_exp@qtr + is@admin_exp@qtr + is@fin_exp@qtr) / sales@qtr' , date)

class roic_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度EBIT(1-税率)/投入资本'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('roic@qtr' , date)

class roic_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM,EBIT(1-税率)/投入资本'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('roic@ttm' , date)
    
class roic_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM,EBIT(1-税率)/投入资本,Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('roic@qtr' , date)

class ebit_tangible_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM,EBIT/有形资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('ebit@ttm / tangible_asset@ttm' , date)
class tax_equ_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度所得税/净资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('tax@qtr / equ@qtr' , date)
    
class tax_equ_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM所得税/净资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('tax@ttm / equ@ttm' , date)

class tax_equ_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM所得税/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        return get_hist_zscore('tax@qtr / equ@qtr' , date)