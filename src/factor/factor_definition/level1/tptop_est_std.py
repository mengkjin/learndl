import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tptop_est_std(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '预期利润总额收益率波动率'
    
    def calc_factor(self, date: int):
        """计算预期利润总额收益率的波动率
        1. 获取过去一段时间的预期利润总额和市值数据
        2. 计算比值的标准差
        """
        # TODO: 需要定义获取财务数据的接口
        start_date = TradeDate(date) - 60  # 约3个月的交易日
        total_profit_est = TRADE_DATA.get_consensus_forecast_series(start_date, date, 'total_profit')
        market_value = TRADE_DATA.get_mkt_value_series(start_date, date)
        
        ratio = total_profit_est / market_value
        std = ratio.std()
        return std.rename('factor_value').to_frame() 