import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class volar_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内波动率'
    
    def calc_factor(self, date: int):
        """计算日内波动率
        1. 获取日内价格数据
        2. 计算波动率
        """
        # TODO: 需要定义获取日内数据的接口
        price = TRADE_DATA.get_intraday_price(date)
        
        def calc_volatility(p):
            # 计算日内波动率
            returns = np.log(p / p.shift(1))
            return returns.std() * np.sqrt(240)  # 年化
            
        vol = price.groupby('secid').apply(calc_volatility)
        return vol.rename('factor_value').to_frame() 