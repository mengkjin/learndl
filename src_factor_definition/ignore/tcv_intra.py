import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tcv_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内成交量集中度'
    
    def calc_factor(self, date: int):
        """计算日内成交量集中度
        1. 获取日内成交量数据
        2. 计算集中度指标
        """
        # TODO: 需要定义获取日内数据的接口
        volume = TRADE_DATA.get_intraday_volume(date)
        
        def calc_concentration(v):
            # 计算成交量集中度
            # 例如: 最大成交量占比
            return v.max() / v.sum()
            
        concentration = volume.groupby('secid').apply(calc_concentration)
        return concentration.rename('factor_value').to_frame() 