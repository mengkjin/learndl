import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class vardown_intra5min(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '下行波动占比'
    
    def calc_factor(self, date: int):
        """计算5分钟下行波动占比
        1. 获取5分钟频率收益率数据
        2. 计算下行波动占总波动的比例
        """
        # TODO: 需要定义获取日内数据的接口
        ret_5min = TRADE_DATA.get_intraday_returns(date, freq='5min')
        
        def calc_downside_var_ratio(x):
            x = pd.Series(x)
            down_var = x[x < 0].var()
            total_var = x.var()
            return down_var / total_var if total_var != 0 else np.nan
            
        var_ratio = ret_5min.apply(calc_downside_var_ratio)
        return var_ratio.rename('factor_value').to_frame() 