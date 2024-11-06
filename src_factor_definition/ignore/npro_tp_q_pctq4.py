import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_tp_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额同比变化率'
    
    def calc_factor(self, date: int):
        """计算季度利润总额同比增长率
        1. 获取当前和去年同期的季度利润总额数据
        2. 计算同比变化率
        """
        # TODO: 需要定义获取财务数据的接口
        tp_curr = TRADE_DATA.get_income_statement(date, 'total_profit', 'quarterly')
        tp_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-4, 'Q'), 'total_profit', 'quarterly')
        
        growth_rate = (tp_curr - tp_prev) / abs(tp_prev)
        return growth_rate.rename('factor_value').to_frame() 