import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class amihud(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'liquidity'
    description = 'Amihud非流动性'
    
    def calc_factor(self, date: int):
        """计算Amihud非流动性指标
        1. 获取过去一段时间的收益率和成交额数据
        2. 计算|收益率|/成交额的平均值
        """
        # TODO: 需要定义获取价格和成交额数据的接口
        start_date = TradeDate(date) - 20  # 约1个月的交易日
        returns = TRADE_DATA.get_returns(start_date, date)
        volume = TRADE_DATA.get_volume(start_date, date)
        amount = TRADE_DATA.get_amount(start_date, date)
        
        illiq = (abs(returns) / amount).mean()
        return illiq.rename('factor_value').to_frame() 