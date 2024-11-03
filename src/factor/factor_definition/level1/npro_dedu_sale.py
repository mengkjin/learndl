import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_dedu_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '扣非归母利润/营业收入'
    
    def calc_factor(self, date: int):
        """计算扣非净利润收入比
        1. 获取扣非净利润和营业收入数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        net_profit_deducted = TRADE_DATA.get_income_statement(date, 'net_profit_deducted')
        revenue = TRADE_DATA.get_income_statement(date, 'revenue')
        
        ratio = net_profit_deducted / revenue
        return ratio.rename('factor_value').to_frame() 