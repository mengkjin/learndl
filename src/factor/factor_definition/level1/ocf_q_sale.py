import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ocf_q_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '经营活动产生的现金流量季度/营业收入'
    
    def calc_factor(self, date: int):
        """计算季度经营现金流收入比
        1. 获取季度经营现金流和营业收入数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        op_cashflow = TRADE_DATA.get_cashflow(date, 'operating_cashflow', 'quarterly')
        revenue = TRADE_DATA.get_income_statement(date, 'revenue', 'quarterly')
        
        ratio = op_cashflow / revenue
        return ratio.rename('factor_value').to_frame() 