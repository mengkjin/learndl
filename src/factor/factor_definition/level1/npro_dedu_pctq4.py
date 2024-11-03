import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_dedu_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '扣非净利润同比增长率'
    
    def calc_factor(self, date: int):
        """计算扣非净利润同比增长率
        1. 获取当期和去年同期的扣非净利润数据
        2. 计算同比增长率
        """
        # TODO: 需要定义获取财务数据的接口
        curr_np = TRADE_DATA.get_income_statement(date, 'net_profit_deducted')
        prev_np = TRADE_DATA.get_income_statement(TradeDate(date).offset(-4, 'Q'), 'net_profit_deducted')
        
        pct_change = (curr_np - prev_np) / abs(prev_np)
        return pct_change.rename('factor_value').to_frame() 