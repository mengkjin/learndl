import numpy as np
import pandas as pd

from typing import Literal
from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


def cap_classic(date , cap_type : Literal['tt' , 'fl' , 'fr']):
    val = DATAVENDOR.TRADE.get_val(date).set_index('secid')
    if cap_type == 'tt':
        return np.log(val['total_share'] * val['close'])
    elif cap_type == 'fl':
        return np.log(val['float_share'] * val['close'])
    elif cap_type == 'fr':
        return np.log(val['free_share'] * val['close'])
    else:
        raise ValueError(f'cap_type {cap_type} not supported')

class lncap(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '市值(对数总市值)'
    
    def calc_factor(self, date: int):
        return cap_classic(date , 'tt')
    
class lncap_liq(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '市值(对数流通市值)'
    
    def calc_factor(self, date: int):
        return cap_classic(date , 'fl')
    
class lncap_free(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '市值(对数自由流通市值)'
    
    def calc_factor(self, date: int):
        return cap_classic(date , 'fr')

class lockedstk(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '锁仓比'
    
    def calc_factor(self, date: int):
        val = DATAVENDOR.TRADE.get_val(date).set_index('secid')
        locked = 1 - val['free_share'] / val['total_share']
        return locked