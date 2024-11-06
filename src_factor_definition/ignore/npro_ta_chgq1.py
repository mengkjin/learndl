import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_ta_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_ta环比变化值'
    
    def calc_factor(self, date: int):
        """计算净利润资产比的环比变化
        1. 获取当前和上季度的净利润和总资产数据
        2. 计算比率的环比变化
        """
        # TODO: 需要定义获取财务数据的接口
        npro_curr = TRADE_DATA.get_income_statement(date, 'net_profit')
        ta_curr = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        npro_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Q'), 'net_profit')
        ta_prev = TRADE_DATA.get_balance_sheet(TradeDate(date).offset(-1, 'Q'), 'total_assets')
        
        ratio_curr = npro_curr / ta_curr
        ratio_prev = npro_prev / ta_prev
        change = ratio_curr - ratio_prev
        
        return change.rename('factor_value').to_frame() 