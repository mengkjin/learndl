import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tptop_est_pct6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '预期利润总额收益率6个月变化率'
    
    def calc_factor(self, date: int):
        """计算预期利润总额收益率6个月变化率
        1. 获取当期和6个月前的预期利润总额和市值数据
        2. 计算变化率
        """
        # TODO: 需要定义获取财务数据的接口
        curr_profit = TRADE_DATA.get_consensus_forecast(date, 'total_profit')
        curr_mv = TRADE_DATA.get_market_value(date)
        
        prev_date = TradeDate(date).offset(-6, 'M')
        prev_profit = TRADE_DATA.get_consensus_forecast(prev_date, 'total_profit')
        prev_mv = TRADE_DATA.get_market_value(prev_date)
        
        curr_ratio = curr_profit / curr_mv
        prev_ratio = prev_profit / prev_mv
        pct_change = (curr_ratio - prev_ratio) / abs(prev_ratio)
        
        return pct_change.rename('factor_value').to_frame()