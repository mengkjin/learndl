import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class amihud_trend(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'liquidity'
    description = 'Amihud非流动性趋势'
    
    def calc_factor(self, date: int):
        """计算Amihud非流动性指标的趋势
        1. 获取过去一段时间的收益率和成交额数据
        2. 计算|收益率|/成交额的趋势
        """
        # TODO: 需要定义获取价格和成交额数据的接口
        start_date = TradeDate(date) - 60  # 约3个月的交易日
        returns = TRADE_DATA.get_returns(start_date, date)
        volume = TRADE_DATA.get_volume(start_date, date)
        amount = TRADE_DATA.get_amount(start_date, date)
        
        illiq = abs(returns) / amount
        
        def calc_trend(x):
            x = pd.Series(x)
            days = range(len(x))
            slope = np.polyfit(days, x, 1)[0]
            return slope
            
        trend = illiq.apply(calc_trend)
        return trend.rename('factor_value').to_frame() 