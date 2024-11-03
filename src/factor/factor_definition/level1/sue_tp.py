import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class sue_tp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '未预期-利润总额'
    
    def calc_factor(self, date: int):
        """计算利润总额的未预期因子
        1. 获取实际利润总额和预测利润总额数据
        2. 计算未预期值
        """
        # TODO: 需要定义获取财务数据的接口
        actual_tp = TRADE_DATA.get_income_statement(date, 'total_profit')
        forecast_tp = TRADE_DATA.get_consensus_forecast(date, 'total_profit')
        
        sue = (actual_tp - forecast_tp) / abs(forecast_tp)
        return sue.rename('factor_value').to_frame() 