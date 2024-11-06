import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class turn_utd(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内换手率上行趋势'
    
    def calc_factor(self, date: int):
        """计算日内换手率上行趋势
        1. 获取日内换手率数据
        2. 计算上升趋势占比
        """
        # TODO: 需要定义获取日内数据的接口
        turnover = TRADE_DATA.get_intraday_turnover(date)
        
        def calc_up_trend(t):
            # 计算上升趋势占比
            diff = t.diff()
            up_periods = (diff > 0).sum()
            total_periods = len(diff) - 1  # 减去第一个nan
            return up_periods / total_periods
            
        utd = turnover.groupby('secid').apply(calc_up_trend)
        return utd.rename('factor_value').to_frame() 