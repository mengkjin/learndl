import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class mom_3m_0m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'momentum'
    description = '3个月动量(无间隔)'
    
    def calc_factor(self, date: int):
        """计算3个月动量(无间隔)
        1. 获取过去3个月的收益率数据
        2. 计算累积收益率
        """
        # TODO: 需要定义获取收益率数据的接口
        start_date = TradeDate(date) - 60  # 约3个月的交易日
        returns = TRADE_DATA.get_returns(start_date, date)
        
        mom = (1 + returns).prod() - 1
        return mom.rename('factor_value').to_frame() 