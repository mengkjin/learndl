import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'btop_est' , 'btop_est_pct3m' , 'btop_est_pct6m' ,
    'etop_est' , 'etop_est_pct3m' , 'etop_est_pct6m' ,
    'npro_pct3m_est' , 'npro_pct6m_est' , 'npro_pctfttm' ,
    'optop_est' , 'optop_est_pct3m' , 'optop_est_pct6m' ,
    'peg_est' , 'stop_est' , 'stop_est_pct3m' , 'stop_est_pct6m' ,
    'tptop_est' , 'tptop_est_pct3m' , 'tptop_est_pct6m'
]

class btop_est(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '净资产/市值一致预期值'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'btop_est')
        return df


class btop_est_pct3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月净资产/市值一致预期变化'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'btop_est_pct3m')
        return df

class btop_est_pct6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月净资产/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'btop_est_pct6m')
        return df

class etop_est(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '归母净利润/市值一致预期值'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'etop_est')
        return df
    
class etop_est_pct3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月归母净利润/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'etop_est_pct3m')
        return df
    
class etop_est_pct6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月归母净利润/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'etop_est_pct6m')
        return df

class npro_pct3m_est(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月净利润/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'npro_pct3m_est')
        return df
    
class npro_pct6m_est(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月净利润/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'npro_pct6m_est')
        return df

class npro_pctfttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '净利润预期/净利润FY0一致预期值'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'npro_pctfttm')
        return df

class optop_est(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '营业利润/市值一致预期值'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'optop_est')
        return df
    
class optop_est_pct3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月营业利润/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'optop_est_pct3m')
        return df   
    
class optop_est_pct6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月营业利润/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'optop_est_pct6m')
        return df

class peg_est(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = 'PEG一致预期值'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'peg_est')
        return df

class stop_est(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '营业收入/市值一致预期值'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'stop_est')
        return df
    
class stop_est_pct3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月营业收入/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'stop_est_pct3m')
        return df

class stop_est_pct6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月营业收入/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'stop_est_pct6m')
        return df

class tptop_est(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '利润总额/市值一致预期值'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'tptop_est')
        return df
    
class tptop_est_pct3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月利润总额/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'tptop_est_pct3m')
        return df

class tptop_est_pct6m(StockFactorCalculator):   
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月利润总额/市值一致预期变化'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast' , date , 'tptop_est_pct6m')
        return df
    
