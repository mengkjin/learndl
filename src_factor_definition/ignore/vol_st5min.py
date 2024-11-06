import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class vol_st5min(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '开盘5分钟成交量'
    
    def calc_factor(self, date: int):
        """计算开盘5分钟成交量
        1. 获取日内前5分钟的成交量数据
        2. 计算总和
        """
        # TODO: 需要定义获取日内成交量数据的接口
        start_time = '09:30:00'  # 开盘时间
        volume = TRADE_DATA.get_intraday_volume(date, start_time, minutes=5)
        
        return volume.rename('factor_value').to_frame()