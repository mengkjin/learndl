import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'outperform_title' , 'outperform_titlepct' , 'upnpro_est_qua'
]

class outperform_title(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'surprise'
    description = '研报标题超预期个数'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'outperform_title')
        return df
    
class outperform_titlepct(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'surprise'
    description = '研报标题超预期比例'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'outperform_titlepct')
        return df
    
class upnpro_est_qua(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'surprise'
    description = '单季度净利润超预期幅度'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'upnpro_est_qua')
        return df
