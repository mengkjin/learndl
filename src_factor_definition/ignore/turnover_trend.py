import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class turnover_trend(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'liquidity'
    description = '换手率趋势'
    
    def calc_factor(self, date: int):
        """计算换手率趋势
        1. 获取过去一段时间的换手率数据
        2. 计算趋势特征
        """
        # TODO: 需要定义获取换手率数据的接口
        start_date = TradeDate(date) - 60  # 约3个月的交易日
        turnover = TRADE_DATA.get_turnover(start_date, date)
        
        def calc_trend(x):
            x = pd.Series(x)
            days = range(len(x))
            slope = np.polyfit(days, x, 1)[0]
            return slope
            
        trend = turnover.apply(calc_trend)
        return trend.rename('factor_value').to_frame() 