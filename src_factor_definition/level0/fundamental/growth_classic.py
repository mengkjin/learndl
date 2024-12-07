import numpy as np
import pandas as pd

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR


class asset_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '总资产TTM同比变化率'
    
    def calc_factor(self, date: int):
        ...

class egro_asset(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-总资产'
    
    def calc_factor(self, date: int):
        ...

class egro_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-净资产'
    
    def calc_factor(self, date: int):
        ...

class egro_sales(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-营业收入'
    
    def calc_factor(self, date: int):
        ...

class egro_npro(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-归母净利润'
    
    def calc_factor(self, date: int):
        ...

class equ_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '净资产同比变化率'
    
    def calc_factor(self, date: int):
        ...

class gp_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润加速度'
    
    def calc_factor(self, date: int):
        ...

class gp_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'gp单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...

class gp_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润同比变化率'
    
    def calc_factor(self, date: int):
        ...


class gp_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润同比变化率'
    
    def calc_factor(self, date: int):
        ...

class gp_q_ta_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'gp_q_ta同比变化值'
    
    def calc_factor(self, date: int):
        ...


class gp_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class gp_ta_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'gp_ta环比变化值'
    
    def calc_factor(self, date: int):
        ...

class gp_ta_trendhb(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润增长趋势_环比'
    
    def calc_factor(self, date: int):
        ...


class gp_ta_trendtb(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润增长趋势_同比'
    
    def calc_factor(self, date: int):
        ...


class liab_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '总负债同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '归母净利润加速度'
    
    def calc_factor(self, date: int):
        ...


class npro_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...

class npro_dedu_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '扣非归母净利润加速度'
    
    def calc_factor(self, date: int):
        ...


class npro_dedu_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_dedu_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...

class npro_dedu_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '扣非归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...


class npro_op_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业利润加速度'
    
    def calc_factor(self, date: int):
        ...


class npro_op_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_op单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...

class npro_op_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业利润同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_op_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业利润同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_op_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class npro_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_q_equ_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_q_equ同比变化值'
    
    def calc_factor(self, date: int):
        ...

class npro_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        ...

class npro_q_ta_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_q_ta同比变化值'
    
    def calc_factor(self, date: int):
        ...


class npro_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class npro_ta_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_ta环比变化值'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额加速度'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_tp单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class npro_trend(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '业绩趋势因子'
    
    def calc_factor(self, date: int):
        ...


class op_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'op_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...


class op_q_equ_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'op_q_equ同比变化值'
    
    def calc_factor(self, date: int):
        ...


class sales_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入加速度'
    
    def calc_factor(self, date: int):
        ...


class sales_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'sales单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...


class sales_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入同比变化率'
    
    def calc_factor(self, date: int):
        ...


class sales_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入同比变化率'
    
    def calc_factor(self, date: int):
        ...


class sales_q_ta_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'sale_q_ta同比变化值'
    
    def calc_factor(self, date: int):
        ...


class sales_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class sales_ta_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'sale_ta环比变化值'
    
    def calc_factor(self, date: int):
        ...


class tax_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税加速度'
    
    def calc_factor(self, date: int):
        ...


class tax_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tax单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...

class tax_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tax_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...


class tax_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税同比变化率'
    
    def calc_factor(self, date: int):
        ...


class tax_q_equ_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tax_q_equ同比变化值'
    
    def calc_factor(self, date: int):
        ...


class tax_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税同比变化率'
    
    def calc_factor(self, date: int):
        ...

class tax_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class tp_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tp_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...


class tp_q_equ_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tp_q_equ同比变化值'
    
    def calc_factor(self, date: int):
        ...
