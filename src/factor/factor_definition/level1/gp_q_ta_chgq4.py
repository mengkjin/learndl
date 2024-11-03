import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class gp_q_ta_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '季度毛利润资产比环比变化'
    
    def calc_factor(self, date: int):
        """计算季度毛利润资产比环比变化
        1. 获取当季和上季的毛利润和总资产数据
        2. 计算环比变化
        """
        # TODO: 需要定义获取财务数据的接口
        curr_gp = TRADE_DATA.get_income_statement(date, 'gross_profit', 'quarterly')
        curr_ta = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        prev_date = TradeDate(date).offset(-1, 'Q')
        prev_gp = TRADE_DATA.get_income_statement(prev_date, 'gross_profit', 'quarterly')
        prev_ta = TRADE_DATA.get_balance_sheet(prev_date, 'total_assets')
        
        curr_ratio = curr_gp / curr_ta
        prev_ratio = prev_gp / prev_ta
        chg = (curr_ratio - prev_ratio) / abs(prev_ratio)
        
        return chg.rename('factor_value').to_frame() 