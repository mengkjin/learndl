import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class vol_stdwei(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '成交量加权标准差占比'
    
    def calc_factor(self, date: int):
        """计算成交量加权标准差占比
        1. 获取日内分钟级别收益率和成交量数据
        2. 计算成交量加权的收益率标准差
        """
        # TODO: 需要定义获取日内数据的接口
        ret = TRADE_DATA.get_intraday_returns(date)
        volume = TRADE_DATA.get_intraday_volume(date)
        
        def calc_vol_weighted_std(r, v):
            weights = v / v.sum()
            weighted_mean = (r * weights).sum()
            weighted_var = (weights * (r - weighted_mean)**2).sum()
            return np.sqrt(weighted_var)
            
        vol_std = pd.DataFrame({'ret': ret, 'volume': volume}).apply(
            lambda x: calc_vol_weighted_std(x['ret'], x['volume']), axis=1)
        return vol_std.rename('factor_value').to_frame() 