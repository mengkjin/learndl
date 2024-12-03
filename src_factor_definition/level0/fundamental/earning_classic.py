import pandas as pd
import numpy as np

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

class gp_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM毛利润除以营业收入'
    
    def calc_factor(self, date: int):
        ...

class gp_sale_q(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度毛利润除以营业收入'
    
    def calc_factor(self, date: int):
        ...

class gp_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM毛利润除以总资产'
    
    def calc_factor(self, date: int):
        ...

class gp_q_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度毛利润除以总资产'
    
    def calc_factor(self, date: int):
        ...


class gp_ta_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '毛利润/总资产,Z-Score'
    
    def calc_factor(self, date: int):
        ...

class lpnp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '线性提纯净利润'
    
    def calc_factor(self, date: int):
        ...

class npro_cratio(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '核心净利润占比'
    
    def calc_factor(self, date: int):
        ...

class npro_dedu_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '扣非归母净利润/净资产'
    
    def calc_factor(self, date: int):
        ...

class npro_dedu_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '扣非归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        ...

class npro_dedu_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '扣非归母净利润/总资产'
    
    def calc_factor(self, date: int):
        ...

class npro_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '归母净利润/净资产'
    
    def calc_factor(self, date: int):
        ...

class npro_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '归母净利润/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        ...

class npro_q_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/净资产'
    
    def calc_factor(self, date: int):
        ...

class npro_q_stk(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/实收资本'
    
    def calc_factor(self, date: int):
        ...

class npro_q_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/总资产'
    
    def calc_factor(self, date: int):
        ...

class npro_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        ...

class npro_q_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度归母净利润/营业收入'
    
    def calc_factor(self, date: int):
        ...

class npro_stk(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/实收资本'
    
    def calc_factor(self, date: int):
        ...

class npro_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/总资产'
    
    def calc_factor(self, date: int):
        ...

class npro_ta_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM归母净利润/总资产,Z-Score'
    
    def calc_factor(self, date: int):
        ...

class ocf_q_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度经营活动现金流/营业收入'
    
    def calc_factor(self, date: int):
        ...

class ocf_q_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度经营活动现金流/总资产'
    
    def calc_factor(self, date: int):
        ...

class ocf_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/营业收入'
    
    def calc_factor(self, date: int):
        ...

class ocf_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM经营活动现金流/总资产'
    
    def calc_factor(self, date: int):
        ...

class ocfa(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '产能利用率提升'
    
    def calc_factor(self, date: int):
        ...

class op_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM营业利润/净资产'
    
    def calc_factor(self, date: int):
        ...

class op_equ_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM营业利润/净资产,Z-Score'
    
    def calc_factor(self, date: int):
        ...

class op_q_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '单季度营业利润/净资产'
    
    def calc_factor(self, date: int):
        ...

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
