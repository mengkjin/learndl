import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_pctfttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '滚动利润预测/滚动利润FY0-一致预测值'
    
    def calc_factor(self, date: int):
        """计算滚动利润预测相对一致预测的偏离
        1. 获取滚动利润预测
        2. 获取一致预测值
        3. 计算偏离度
        """
        # TODO: 需要定义获取分析师预测数据的接口
        rolling_est = TRADE_DATA.get_rolling_npro_est(date)
        consensus_est = TRADE_DATA.get_consensus_npro_est(date)
        
        pct = (rolling_est - consensus_est) / abs(consensus_est)
        return pct.rename('factor_value').to_frame() 