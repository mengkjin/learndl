import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tp_sale_q(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '利润总额/营业收入'
    
    def calc_factor(self, date: int):
        """计算季度利润总额收入比
        1. 获取季度利润总额和营业收入数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        total_profit = TRADE_DATA.get_income_statement(date, 'total_profit', 'quarterly')
        revenue = TRADE_DATA.get_income_statement(date, 'revenue', 'quarterly')
        
        ratio = total_profit / revenue
        return ratio.rename('factor_value').to_frame() 