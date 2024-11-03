import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class gp_ta_trendhb(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润资产比趋势环比'
    
    def calc_factor(self, date: int):
        """计算毛利润资产比趋势环比
        1. 获取最近几期的毛利润和总资产数据
        2. 计算趋势的环比变化
        """
        # TODO: 需要定义获取财务数据的接口
        # 获取最近4个季度的数据
        quarters = [TradeDate(date).offset(-i, 'Q') for i in range(4)]
        
        ratios = []
        for q in quarters:
            gp = TRADE_DATA.get_income_statement(q, 'gross_profit')
            ta = TRADE_DATA.get_balance_sheet(q, 'total_assets')
            ratios.append(gp / ta)
            
        # 计算趋势
        def calc_trend(x):
            days = range(len(x))
            slope = np.polyfit(days, x, 1)[0]
            return slope
            
        # 计算环比变化
        curr_trend = calc_trend(ratios[:2])  # 当期趋势
        prev_trend = calc_trend(ratios[1:3])  # 上期趋势
        trend_chg = (curr_trend - prev_trend) / abs(prev_trend)
        
        return trend_chg.rename('factor_value').to_frame()