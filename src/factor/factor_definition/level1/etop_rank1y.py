import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class etop_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '盈利收益率分位数'
    
    def calc_factor(self, date: int):
        """计算盈利收益率的1年分位数
        1. 获取净利润和市值数据
        2. 计算比值的历史分位数
        """
        # TODO: 需要定义获取财务数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        net_profit = TRADE_DATA.get_profit_ttm(date)
        market_value = TRADE_DATA.get_mkt_value(date)
        
        ratio = net_profit / market_value
        return ratio.rename('factor_value').to_frame() 