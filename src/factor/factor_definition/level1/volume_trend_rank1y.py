import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class volume_trend_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'sentiment'
    description = '成交量趋势分位数'
    
    def calc_factor(self, date: int):
        """计算成交量趋势的1年分位数
        1. 获取过去1年的成交量数据
        2. 计算趋势的历史分位数
        """
        # TODO: 需要定义获取成交量数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        volume = TRADE_DATA.get_volume(start_date, date)
        
        def calc_trend(x):
            x = pd.Series(x)
            days = range(len(x))
            slope = np.polyfit(days, x, 1)[0]
            return slope
            
        trend = volume.apply(calc_trend)
        rank = trend.rank(pct=True)
        return rank.rename('factor_value').to_frame() 