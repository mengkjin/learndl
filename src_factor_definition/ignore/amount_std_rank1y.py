import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class amount_std_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'sentiment'
    description = '成交额波动率分位数'
    
    def calc_factor(self, date: int):
        """计算成交额波动率的1年分位数
        1. 获取过去1年的成交额数据
        2. 计算波动率的历史分位数
        """
        # TODO: 需要定义获取成交额数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        amount = TRADE_DATA.get_amount(start_date, date)
        
        # 计算滚动波动率
        def calc_rolling_std(x):
            return pd.Series(x).rolling(20).std()
            
        vol = amount.apply(calc_rolling_std)
        rank = vol.rank(pct=True)
        return rank.rename('factor_value').to_frame() 