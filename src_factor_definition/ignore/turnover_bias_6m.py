import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class turnover_bias_6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'sentiment'
    description = '6个月换手率乖离率'
    
    def calc_factor(self, date: int):
        """计算6个月换手率乖离率
        1. 获取过去6个月的换手率数据
        2. 计算当前值与均值的偏离程度
        """
        # TODO: 需要定义获取换手率数据的接口
        start_date = TradeDate(date) - 120  # 约6个月的交易日
        turnover = TRADE_DATA.get_turnover(start_date, date)
        
        # 计算乖离率
        current = turnover.iloc[-1]
        mean = turnover.mean()
        bias = (current - mean) / mean
        
        return bias.rename('factor_value').to_frame() 