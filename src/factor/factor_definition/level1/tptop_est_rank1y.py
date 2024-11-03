import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tptop_est_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '预期利润总额收益率分位数'
    
    def calc_factor(self, date: int):
        """计算预期利润总额收益率的1年分位数
        1. 获取预期利润总额和市值数据
        2. 计算比值的历史分位数
        """
        # TODO: 需要定义获取财务数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        total_profit_est = TRADE_DATA.get_consensus_forecast(date, 'total_profit')
        market_value = TRADE_DATA.get_mkt_value(date)
        
        ratio = total_profit_est / market_value
        return ratio.rename('factor_value').to_frame() 