import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_pct6m_est(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '盈利指标变化值'
    
    def calc_factor(self, date: int):
        """计算6个月盈利预测变化
        1. 获取当前分析师预测
        2. 获取6个月前分析师预测
        3. 计算变化率
        """
        # TODO: 需要定义获取分析师预测数据的接口
        current_est = TRADE_DATA.get_analyst_est_npro(date)
        prev_est = TRADE_DATA.get_analyst_est_npro(TradeDate(date) - 120)
        
        pct_change = (current_est - prev_est) / abs(prev_est)
        return pct_change.rename('factor_value').to_frame() 