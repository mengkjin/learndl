import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ocfa(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '经营活动现金流量'
    
    def calc_factor(self, date: int):
        """获取经营活动现金流量
        1. 获取经营活动现金流量数据
        """
        # TODO: 需要定义获取财务数据的接口
        operating_cashflow = TRADE_DATA.get_cashflow(date, 'operating_cashflow')
        return operating_cashflow.rename('factor_value').to_frame() 