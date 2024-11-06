import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class apm_orig(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '原始Amihud价格冲击'
    
    def calc_factor(self, date: int):
        """计算原始Amihud价格冲击
        1. 获取日内收益率和成交额数据
        2. 计算价格冲击指标
        """
        # TODO: 需要定义获取日内数据的接口
        returns = TRADE_DATA.get_intraday_returns(date)
        amount = TRADE_DATA.get_intraday_amount(date)
        
        def calc_apm(r, a):
            # 计算价格冲击
            return abs(r).mean() / a.mean()
            
        apm = returns.groupby('secid').apply(lambda x: calc_apm(x, amount[x.index]))
        return apm.rename('factor_value').to_frame() 