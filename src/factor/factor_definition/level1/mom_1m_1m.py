import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class mom_1m_1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'momentum'
    description = '1个月动量(间隔1个月)'
    
    def calc_factor(self, date: int):
        """计算1个月动量(间隔1个月)
        1. 获取过去1-2个月的收益率数据
        2. 计算累积收益率
        """
        # TODO: 需要定义获取收益率数据的接口
        end_date = TradeDate(date) - 20  # 间隔1个月
        start_date = end_date - 20  # 再往前1个月
        returns = TRADE_DATA.get_returns(start_date, end_date)
        
        mom = (1 + returns).prod() - 1
        return mom.rename('factor_value').to_frame() 