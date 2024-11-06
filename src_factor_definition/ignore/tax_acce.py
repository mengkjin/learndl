import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tax_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税加速度'
    
    def calc_factor(self, date: int):
        """计算所得税加速度
        1. 获取最近几个季度的所得税数据
        2. 计算环比增速的变化率
        """
        # TODO: 需要定义获取财务数据的接口
        tax_q = TRADE_DATA.get_income_statement(date, 'income_tax', 'quarterly')
        tax_q_1 = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Q'), 'income_tax', 'quarterly')
        tax_q_2 = TRADE_DATA.get_income_statement(TradeDate(date).offset(-2, 'Q'), 'income_tax', 'quarterly')
        
        # 计算两个相邻季度的增速变化
        growth_curr = (tax_q - tax_q_1) / abs(tax_q_1)
        growth_prev = (tax_q_1 - tax_q_2) / abs(tax_q_2)
        acceleration = growth_curr - growth_prev
        
        return acceleration.rename('factor_value').to_frame()