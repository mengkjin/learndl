import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class mkt_size(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'size'
    description = '市值规模'
    
    def calc_factor(self, date: int):
        """计算市值规模
        1. 获取市值数据
        2. 取对数
        """
        # TODO: 需要定义获取市值数据的接口
        market_value = TRADE_DATA.get_market_value(date)
        
        log_mkt = np.log(market_value)
        return log_mkt.rename('factor_value').to_frame() 