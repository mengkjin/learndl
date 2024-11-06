import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class vol_vwapcorr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_correlation'
    description = '成交量价格相关'
    
    def calc_factor(self, date: int):
        """计算成交量与VWAP的相关系数
        1. 获取日内分钟级别成交量和VWAP数据
        2. 计算相关系数
        """
        # TODO: 需要定义获取日内数据的接口
        volume = TRADE_DATA.get_intraday_volume(date)
        vwap = TRADE_DATA.get_intraday_vwap(date)
        
        corr = pd.DataFrame({'volume': volume, 'vwap': vwap}).apply(
            lambda x: x['volume'].corr(x['vwap']), axis=1)
        return corr.rename('factor_value').to_frame() 