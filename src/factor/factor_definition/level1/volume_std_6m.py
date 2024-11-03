import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class volume_std_6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'sentiment'
    description = '6个月成交量波动率'
    
    def calc_factor(self, date: int):
        """计算6个月成交量波动率
        1. 获取过去6个月的成交量数据
        2. 计算标准差
        """
        # TODO: 需要定义获取成交量数据的接口
        start_date = TradeDate(date) - 120  # 约6个月的交易日
        volume = TRADE_DATA.get_volume(start_date, date)
        
        std = volume.std()
        return std.rename('factor_value').to_frame() 