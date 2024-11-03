import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ocftop_q_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'ocftop_q分位数'
    
    def calc_factor(self, date: int):
        """计算季度经营现金流市值比的1年分位数
        1. 获取过去1年的季度经营现金流和市值数据
        2. 计算当前值在历史分布中的分位数
        """
        # TODO: 需要定义获取财务数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        op_cashflow = TRADE_DATA.get_cashflow(date, 'operating_cashflow', 'quarterly')
        market_value = TRADE_DATA.get_mkt_value(date)
        
        ratio = op_cashflow / market_value
        return ratio.rename('factor_value').to_frame() 