import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class gp_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润加速度'
    
    def calc_factor(self, date: int):
        """计算毛利润加速度
        1. 获取最近三期的毛利润数据
        2. 计算加速度
        """
        # TODO: 需要定义获取财务数据的接口
        gp_t = TRADE_DATA.get_income_statement(date, 'gross_profit')
        gp_t1 = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Q'), 'gross_profit')
        gp_t2 = TRADE_DATA.get_income_statement(TradeDate(date).offset(-2, 'Q'), 'gross_profit')
        
        # 计算加速度: (当期增长率 - 上期增长率)
        curr_growth = (gp_t - gp_t1) / abs(gp_t1)
        prev_growth = (gp_t1 - gp_t2) / abs(gp_t2)
        acce = curr_growth - prev_growth
        
        return acce.rename('factor_value').to_frame()