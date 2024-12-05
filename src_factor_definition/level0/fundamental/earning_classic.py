import pandas as pd
import numpy as np
import statsmodels.api as sm

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

def get_latest(numerator : str , date : int , **kwargs):
    return DATAVENDOR.get_fin_latest(numerator , date , **kwargs)

def get_hist(numerator : str , date : int , n_last : int = 1 , **kwargs):
    return DATAVENDOR.get_fin_hist(numerator , date , n_last , **kwargs)

def hist_zscore(data : pd.Series | pd.DataFrame):
    if isinstance(data , pd.DataFrame):
        print(data)
        assert data.shape[1] == 1 , 'data must be a single column'
        data = data.iloc[:,0]
    grp = data.groupby('secid')
    return (grp.last() - grp.mean()) / grp.std()

def ts_std(ts : pd.Series | pd.DataFrame):
    return (ts - ts.mean()) / ts.std()

def ts_reg_resid(ts : pd.DataFrame , y_col : str , x_cols : list[str]):
    y = ts_std(ts[y_col])
    x = sm.add_constant(ts[x_cols])
    model = sm.OLS(y , x , missing = 'drop').fit()
    return model.resid.iloc[-1]

class gp_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM毛利润除以营业收入'
    
    def calc_factor(self, date: int):
        return get_latest('indi@ttm@gross_margin' , date , qtr_method = 'diff') / get_latest('is@ttm@revenue' , date)

class gp_sale_q(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度毛利润除以营业收入'
    
    def calc_factor(self, date: int):
        return get_latest('indi@qtr@gross_margin' , date , qtr_method = 'diff') / get_latest('is@qtr@revenue' , date)

class gp_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM毛利润除以总资产'
    
    def calc_factor(self, date: int):
        return get_latest('indi@ttm@gross_margin' , date , qtr_method = 'diff') / get_latest('bs@ttm@total_assets' , date)

class gp_q_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度毛利润除以总资产'
    
    def calc_factor(self, date: int):
        return get_latest('indi@qtr@gross_margin' , date , qtr_method = 'diff') / get_latest('bs@qtr@total_assets' , date)

class gp_ta_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '毛利润/总资产,Z-Score'
    
    def calc_factor(self, date: int):
        gp = get_hist('indi@qtr@gross_margin' , date , 20).iloc[:,0]
        ta = get_hist('bs@qtr@total_assets' , date , 20).iloc[:,0]
        return hist_zscore(gp / ta)

class lpnp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '线性提纯净利润'
    
    def calc_factor(self, date: int):
        npro = get_latest('is@qtr@n_income_attr_p' , date)
        sales = get_latest('is@qtr@revenue' , date)
        bv = get_latest('bs@qtr@total_hldr_eqy_exc_min_int' , date)

        x = sm.add_constant(pd.concat([sales , bv] , axis = 1).reindex(npro.index))
        model = sm.OLS(npro, x , missing = 'drop').fit()
        return (model.resid - model.resid.mean()) / model.resid.std()

class npro_cratio(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '核心净利润占比'
    
    def calc_factor(self, date: int):
        npro = get_latest('is@qtr@n_income_attr_p' , date)
        gp = get_latest('indi@qtr@gross_margin' , date , qtr_method = 'diff')
        biz_tax_surchg = get_latest('is@qtr@biz_tax_surchg' , date)
        sell_exp = get_latest('is@qtr@sell_exp' , date)
        admin_exp = get_latest('is@qtr@admin_exp' , date)
        fin_exp = get_latest('is@qtr@fin_exp' , date)
        rd = get_latest('is@qtr@rd_exp' , date)
        return (gp - biz_tax_surchg - sell_exp - admin_exp - fin_exp - rd) / npro

class npro_dedu_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '扣非归母净利润/净资产'
    
    def calc_factor(self, date: int):
        npro_dedu = get_latest('indi@ttm@profit_dedt' , date , qtr_method = 'diff')
        bv = get_latest('bs@ttm@total_hldr_eqy_exc_min_int' , date)
        return npro_dedu / bv

class npro_dedu_sales(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '扣非归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        npro_dedu = get_latest('indi@ttm@profit_dedt' , date , qtr_method = 'diff')
        sales = get_latest('is@ttm@revenue' , date)
        return npro_dedu / sales

class npro_dedu_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '扣非归母净利润/总资产'
    
    def calc_factor(self, date: int):
        npro_dedu = get_latest('indi@ttm@profit_dedt' , date , qtr_method = 'diff')
        ta = get_latest('bs@ttm@total_assets' , date)
        return npro_dedu / ta

class npro_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '归母净利润/净资产'
    
    def calc_factor(self, date: int):
        npro = get_latest('is@ttm@n_income_attr_p' , date)
        bv = get_latest('bs@ttm@total_hldr_eqy_exc_min_int' , date)
        return npro / bv

class npro_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '归母净利润/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        npro = get_hist('is@qtr@n_income_attr_p' , date , 20).iloc[:,0]
        bv   = get_hist('bs@qtr@total_hldr_eqy_exc_min_int' , date , 20).iloc[:,0]
        return hist_zscore(npro / bv)
    
class npro_q_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/净资产'
    
    def calc_factor(self, date: int):
        npro = get_latest('is@qtr@n_income_attr_p' , date)
        bv = get_latest('bs@qtr@total_hldr_eqy_exc_min_int' , date)
        return npro / bv

class npro_q_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/总资产'
    
    def calc_factor(self, date: int):
        npro = get_latest('is@qtr@n_income_attr_p' , date)
        ta = get_latest('bs@qtr@total_assets' , date)
        return npro / ta

class npro_sales(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        npro = get_latest('is@ttm@n_income_attr_p' , date)
        sales = get_latest('is@ttm@revenue' , date)
        return npro / sales

class npro_q_sales(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        npro = get_latest('is@qtr@n_income_attr_p' , date)
        sales = get_latest('is@qtr@revenue' , date)
        return npro / sales

class npro_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/总资产'
    
    def calc_factor(self, date: int):
        npro = get_latest('is@ttm@n_income_attr_p' , date)
        ta = get_latest('bs@ttm@total_assets' , date)
        return npro / ta

class npro_ta_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/总资产,Z-Score'
    
    def calc_factor(self, date: int):
        npro = get_hist('is@qtr@n_income_attr_p' , date , 20).iloc[:,0]
        ta   = get_hist('bs@qtr@total_assets' , date , 20).iloc[:,0]
        return hist_zscore(npro / ta)

class ocf_q_sales(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度经营活动现金流/营业收入'
    
    def calc_factor(self, date: int):
        ocf = get_latest('cf@qtr@n_cashflow_act' , date)
        sales = get_latest('is@qtr@revenue' , date)
        return ocf / sales

class ocf_q_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度经营活动现金流/总资产'
    
    def calc_factor(self, date: int):
        ocf = get_latest('cf@qtr@n_cashflow_act' , date)
        ta = get_latest('bs@qtr@total_assets' , date)
        return ocf / ta

class ocf_sales(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/营业收入'
    
    def calc_factor(self, date: int):
        ocf = get_latest('cf@ttm@n_cashflow_act' , date)
        sales = get_latest('is@ttm@revenue' , date)
        return ocf / sales

class ocf_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/总资产'
    
    def calc_factor(self, date: int):
        ocf = get_latest('cf@ttm@n_cashflow_act' , date)
        ta = get_latest('bs@ttm@total_assets' , date)
        return ocf / ta

class ocfa(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'Operation Cost on Fixed Assets,产能利用率提升'
    
    def calc_factor(self, date: int):
        ...

class op_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM营业利润/净资产'
    
    def calc_factor(self, date: int):
        op = get_latest('is@ttm@operate_profit' , date)
        bv = get_latest('bs@ttm@total_hldr_eqy_exc_min_int' , date)
        return op / bv

class op_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM营业利润/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        op = get_hist('is@qtr@operate_profit' , date , 20).iloc[:,0]
        bv = get_hist('bs@qtr@total_hldr_eqy_exc_min_int' , date , 20).iloc[:,0]
        return hist_zscore(op / bv)

class op_q_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度营业利润/净资产'
    
    def calc_factor(self, date: int):
        op = get_latest('is@qtr@operate_profit' , date)
        bv = get_latest('bs@qtr@total_hldr_eqy_exc_min_int' , date)
        return op / bv

class periodexp_q_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度费用/营业收入'
    
    def calc_factor(self, date: int):
        ...

class periodexp_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM费用/营业收入'
    
    def calc_factor(self, date: int):
        ...

class roic(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM,EBIT(1-税率)/投入资本'
    
    def calc_factor(self, date: int):
        ...

class rotc(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM,EBIT/有形资产'
    
    def calc_factor(self, date: int):
        ...

class rroc(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '营业能力改善'
    
    def calc_factor(self, date: int):
        ...

class sales_ta_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '营业收入/总资产,Z-Score'
    
    def calc_factor(self, date: int):
        ...

class tax_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM所得税/净资产'
    
    def calc_factor(self, date: int):
        ...

class tax_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM所得税/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        ...

class tax_q_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度所得税/净资产'
    
    def calc_factor(self, date: int):
        ...

class tp_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM利润总额/净资产'
    
    def calc_factor(self, date: int):
        ...

class tp_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM利润总额/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        ...

class tp_q_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度利润总额/净资产'
    
    def calc_factor(self, date: int):
        ...
