import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tptop_est_pct3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '利润总额/市值预测值3个月变化率'
    
    def calc_factor(self, date: int):
        """计算预测利润总额/市值比率的3个月变化率
        1. 获取当前和3个月前的预测数据
        2. 计算变化率
        """
        # TODO: 需要定义获取财务数据的接口
        curr_est = TRADE_DATA.get_consensus_forecast(date, 'total_profit')
        prev_est = TRADE_DATA.get_consensus_forecast(TradeDate(date).offset(-3, 'M'), 'total_profit')
        
        pct_change = (curr_est - prev_est) / abs(prev_est)
        return pct_change.rename('factor_value').to_frame() 