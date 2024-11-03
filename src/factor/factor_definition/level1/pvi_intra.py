import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class pvi_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内价格波动指标'
    
    def calc_factor(self, date: int):
        """计算日内价格波动指标
        1. 获取日内价格数据
        2. 计算波动指标
        """
        # TODO: 需要定义获取日内数据的接口
        price = TRADE_DATA.get_intraday_price(date)
        
        def calc_pvi(p):
            # 计算价格波动指标
            # 例如: 日内最高价与最低价之差除以均价
            high = p.max()
            low = p.min()
            mean = p.mean()
            return (high - low) / mean
            
        pvi = price.groupby('secid').apply(calc_pvi)
        return pvi.rename('factor_value').to_frame() 