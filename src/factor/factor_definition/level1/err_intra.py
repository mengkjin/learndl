import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class err_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内收益率残差'
    
    def calc_factor(self, date: int):
        """计算日内收益率残差
        1. 获取日内收益率和市场收益率数据
        2. 计算残差
        """
        # TODO: 需要定义获取日内数据的接口
        returns = TRADE_DATA.get_intraday_returns(date)
        market_returns = TRADE_DATA.get_intraday_market_returns(date)
        
        def calc_residual(r):
            # 计算残差
            model = np.polyfit(market_returns, r, 1)
            predicted = np.polyval(model, market_returns)
            residual = r - predicted
            return residual.std()  # 使用残差标准差作为指标
            
        err = returns.groupby('secid').apply(calc_residual)
        return err.rename('factor_value').to_frame() 