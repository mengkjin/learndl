import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class hphigh_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内最高价占比'
    
    def calc_factor(self, date: int):
        """计算日内最高价占比
        1. 获取日内价格数据
        2. 计算最高价占比
        """
        # TODO: 需要定义获取日内数据的接口
        price = TRADE_DATA.get_intraday_price(date)
        
        def calc_high_ratio(p):
            # 计算最高价占比
            high = p.max()
            mean = p.mean()
            return high / mean
            
        ratio = price.groupby('secid').apply(calc_high_ratio)
        return ratio.rename('factor_value').to_frame() 