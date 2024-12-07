
import pandas as pd
import numpy as np

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

class assetcur_asset(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '流动资产/总资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('ca_to_assets' , date) / 100
    
class liab_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '产权比率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('debt_to_eqt' , date)
    
class liab_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '资产负债率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('debt_to_assets' , date) / 100
    
class liabcur_liab(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '流动负债/总负债'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('currentdebt_to_debt' , date) / 100
    
class ratio_cash(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '现金比率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('cash_ratio' , date) / 100
    
class ratio_current(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '流动比率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('current_ratio' , date) / 100
    
class ratio_quick(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '速动比率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('quick_ratio' , date) / 100

class sales_ta_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = 'TTM资产周转率'
    
    def calc_factor(self, date: int):
        sales = DATAVENDOR.IS.ttm_latest('revenue' , date)
        ta    = DATAVENDOR.BS.ttm_latest('total_assets' , date)
        return sales / ta

class sales_ta_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '单季度资产周转率'
    
    def calc_factor(self, date: int):
        sales = DATAVENDOR.IS.qtr_latest('revenue' , date)
        ta    = DATAVENDOR.BS.qtr_latest('total_assets' , date)
        return sales / ta
    
class ta_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '权益乘数'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('assets_to_eqt' , date) / 100