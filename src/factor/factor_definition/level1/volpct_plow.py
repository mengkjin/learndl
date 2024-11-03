import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class volpct_plow(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '价格区间成交占比'
    
    def calc_factor(self, date: int):
        """计算低价区间的成交量占比
        1. 获取日内分钟级别价格和成交量数据
        2. 计算低价区间(<25%分位数)的成交量占比
        """
        # TODO: 需要定义获取日内数据的接口
        price = TRADE_DATA.get_intraday_price(date)
        volume = TRADE_DATA.get_intraday_volume(date)
        
        def calc_low_price_vol_ratio(p, v):
            # 定义低价阈值(25分位数)
            low_price_threshold = p.quantile(0.25)
            # 计算低价区间成交量占比
            low_vol = v[p < low_price_threshold].sum()
            total_vol = v.sum()
            return low_vol / total_vol if total_vol != 0 else np.nan
            
        vol_ratio = pd.DataFrame({'price': price, 'volume': volume}).apply(
            lambda x: calc_low_price_vol_ratio(x['price'], x['volume']), axis=1)
        return vol_ratio.rename('factor_value').to_frame() 