import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class outperform_titlepct(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'surprise'
    description = '目标价超预期占比'
    
    def calc_factor(self, date: int):
        """计算目标价超预期占比因子
        1. 统计分析师目标价超过当前价格的比例
        """
        # 使用get_rets计算收益率
        ret = TRADE_DATA.get_returns(TradeDate(date) - 20, date)
        # TODO: 需要补充分析师目标价数据的获取方法
        
        # 临时返回收益率作为因子值
        df = ret.iloc[-1].rename('factor_value').to_frame()
        return df 