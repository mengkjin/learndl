import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class turnover_std_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'liquidity'
    description = '换手率波动率分位数'
    
    def calc_factor(self, date: int):
        """计算换手率波动率的1年分位数
        1. 获取过去1年的换手率数据
        2. 计算波动率的历史分位数
        """
        # TODO: 需要定义获取换手率数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        turnover = TRADE_DATA.get_turnover(start_date, date)
        
        # 计算滚动波动率
        def calc_rolling_std(x):
            return pd.Series(x).rolling(20).std()
            
        vol = turnover.apply(calc_rolling_std)
        rank = vol.rank(pct=True)
        return rank.rename('factor_value').to_frame() 