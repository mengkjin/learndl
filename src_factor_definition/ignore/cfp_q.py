import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class cfp_q(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '季度现金流量/市值'
    
    def calc_factor(self, date: int):
        """计算季度现金流量市值比
        1. 获取季度现金流量和市值数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        cash_flow = TRADE_DATA.get_cashflow(date, 'net_cashflow', 'quarterly')
        market_value = TRADE_DATA.get_market_value(date)
        
        ratio = cash_flow / market_value
        return ratio.rename('factor_value').to_frame() 