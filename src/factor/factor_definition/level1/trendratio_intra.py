import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class trendratio_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内趋势比率'
    
    def calc_factor(self, date: int):
        """计算日内趋势比率
        1. 获取日内价格数据
        2. 计算趋势比率
        """
        # TODO: 需要定义获取日内数据的接口
        price = TRADE_DATA.get_intraday_price(date)
        
        def calc_trend_ratio(p):
            # 计算趋势比率
            # 例如: 上升趋势时段数除以总时段数
            diff = p.diff()
            up_periods = (diff > 0).sum()
            total_periods = len(diff) - 1  # 减去第一个nan
            return up_periods / total_periods
            
        ratio = price.groupby('secid').apply(calc_trend_ratio)
        return ratio.rename('factor_value').to_frame() 