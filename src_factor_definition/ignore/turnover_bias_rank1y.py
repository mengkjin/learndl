import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class turnover_bias_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'sentiment'
    description = '换手率乖离率分位数'
    
    def calc_factor(self, date: int):
        """计算换手率乖离率的1年分位数
        1. 获取过去1年的换手率数据
        2. 计算乖离率的历史分位数
        """
        # TODO: 需要定义获取换手率数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        turnover = TRADE_DATA.get_turnover(start_date, date)
        
        # 计算滚动乖离率
        def calc_rolling_bias(x):
            return (x - x.rolling(20).mean()) / x.rolling(20).mean()
            
        bias = turnover.apply(calc_rolling_bias)
        rank = bias.rank(pct=True)
        
        return rank.rename('factor_value').to_frame() 