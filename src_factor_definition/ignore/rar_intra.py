import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class rar_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内收益率绝对值比率'
    
    def calc_factor(self, date: int):
        """计算日内收益率绝对值比率
        1. 获取日内收益率数据
        2. 计算绝对值比率
        """
        # TODO: 需要定义获取日内数据的接口
        returns = TRADE_DATA.get_intraday_returns(date)
        
        def calc_rar(r):
            # 计算收益率绝对值比率
            # 例如: 上涨时段收益率绝对值之和除以下跌时段收益率绝对值之和
            up_returns = abs(r[r > 0]).sum()
            down_returns = abs(r[r < 0]).sum()
            return up_returns / down_returns if down_returns != 0 else np.nan
            
        rar = returns.groupby('secid').apply(calc_rar)
        return rar.rename('factor_value').to_frame() 