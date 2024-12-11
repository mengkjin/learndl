import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'holding_actwei' , 'holding_actweibu' , 'holding_mkts' , 'holding_num' , 'holding_ratio' ,
    'holding_relawei' , 'holding_relaweibu' , 'holding_shares'
]

class holding_actwei(StockFactorCalculator):
    init_date = 20070101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '基金持股市值占比-市场市值占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('money_flow' , date , 'holding_actwei')
        return df
    
class holding_actweibu(StockFactorCalculator):
    init_date = 20070101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '基金内个股主动权重中位数'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('money_flow' , date , 'holding_actweibu')
        return df
    
class holding_mkts(StockFactorCalculator):
    init_date = 20070101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '持有个股市值'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('money_flow' , date , 'holding_mkts')
        return df
    
class holding_num(StockFactorCalculator):
    init_date = 20070101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '持有个股基金数量'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('money_flow' , date , 'holding_num')
        return df
    
class holding_ratio(StockFactorCalculator):
    init_date = 20070101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '持有个股股数占股本比例'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('money_flow' , date , 'holding_ratio')
        return df
    
class holding_relawei(StockFactorCalculator):
    init_date = 20070101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '基金持股市值占比/市场市值占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('money_flow' , date , 'holding_relawei')
        return df
    
class holding_relaweibu(StockFactorCalculator):
    init_date = 20070101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '基金内个股主动权重中位数'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('money_flow' , date , 'holding_relaweibu')
        return df
    
class holding_shares(StockFactorCalculator):
    init_date = 20070101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '持有个股股票数量'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('money_flow' , date , 'holding_shares')
        return df
    