import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class maxdd_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内最大回撤'
    
    def calc_factor(self, date: int):
        """计算日内最大回撤
        1. 获取日内价格数据
        2. 计算最大回撤
        """
        # TODO: 需要定义获取日内数据的接口
        price = TRADE_DATA.get_intraday_price(date)
        
        def calc_maxdd(p):
            # 计算最大回撤
            cummax = p.cummax()
            drawdown = (cummax - p) / cummax
            return drawdown.max()
            
        maxdd = price.groupby('secid').apply(calc_maxdd)
        return maxdd.rename('factor_value').to_frame()