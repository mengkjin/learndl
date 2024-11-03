import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ncftop_q(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '季度净现金流收益率'
    
    def calc_factor(self, date: int):
        """计算季度净现金流收益率
        1. 获取季度净现金流和市值数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        net_cashflow = TRADE_DATA.get_cashflow(date, 'net_cashflow', 'quarterly')
        market_value = TRADE_DATA.get_mkt_value(date)
        
        ratio = net_cashflow / market_value
        return ratio.rename('factor_value').to_frame() 