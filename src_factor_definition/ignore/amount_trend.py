import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class amount_trend(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'sentiment'
    description = '成交额趋势'
    
    def calc_factor(self, date: int):
        """计算成交额趋势
        1. 获取过去一段时间的成交额数据
        2. 计算趋势特征
        """
        # TODO: 需要定义获取成交额数据的接口
        start_date = TradeDate(date) - 60  # 约3个月的交易日
        amount = TRADE_DATA.get_amount(start_date, date)
        
        def calc_trend(x):
            x = pd.Series(x)
            days = range(len(x))
            slope = np.polyfit(days, x, 1)[0]
            return slope
            
        trend = amount.apply(calc_trend)
        return trend.rename('factor_value').to_frame() 