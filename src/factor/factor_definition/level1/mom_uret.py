import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class mom_uret(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内上涨收益率'
    
    def calc_factor(self, date: int):
        """计算日内上涨收益率
        1. 获取日内收益率数据
        2. 计算上涨时段的收益率
        """
        # TODO: 需要定义获取日内数据的接口
        returns = TRADE_DATA.get_intraday_returns(date)
        
        def calc_up_returns(r):
            # 计算上涨时段的收益率之和
            return r[r > 0].sum()
            
        up_ret = returns.groupby('secid').apply(calc_up_returns)
        return up_ret.rename('factor_value').to_frame() 