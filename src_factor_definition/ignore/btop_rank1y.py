import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class btop_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'btop分位数'
    
    def calc_factor(self, date: int):
        """计算btop的1年分位数
        1. 获取过去1年的btop数据
        2. 计算当前值在历史分布中的分位数
        """
        # TODO: 需要定义获取财务数据的接口
        start_date = TradeDate(date).offset(-12, 'M')
        book_values = TRADE_DATA.get_balance_sheet_series(start_date, date, 'total_equity')
        market_values = TRADE_DATA.get_market_value_series(start_date, date)
        
        btop_series = book_values / market_values
        rank = btop_series.groupby('secid').apply(lambda x: x.rank(pct=True).iloc[-1])
        
        return rank.rename('factor_value').to_frame() 