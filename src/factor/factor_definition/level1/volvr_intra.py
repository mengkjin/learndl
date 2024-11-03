import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class volvr_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '成交量占比'
    
    def calc_factor(self, date: int):
        """计算日内不同时段的成交量占比
        1. 获取日内分钟级别成交量数据
        2. 计算不同时段成交量占比的特征
        """
        # TODO: 需要定义获取日内数据的接口
        volume = TRADE_DATA.get_intraday_volume(date)
        
        def calc_volume_ratio(v):
            # 将交易日分为三段
            n = len(v)
            first_period = v[:n//3].sum()
            middle_period = v[n//3:2*n//3].sum()
            last_period = v[2*n//3:].sum()
            total = v.sum()
            # 返回中间时段占比
            return middle_period / total if total != 0 else np.nan
            
        vol_ratio = volume.apply(calc_volume_ratio)
        return vol_ratio.rename('factor_value').to_frame() 