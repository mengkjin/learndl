import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class sue_op(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '未预期-营业利润'
    
    def calc_factor(self, date: int):
        """计算营业利润的未预期因子
        1. 获取实际营业利润和预测营业利润数据
        2. 计算未预期值
        """
        # TODO: 需要定义获取财务数据的接口
        actual_op = TRADE_DATA.get_income_statement(date, 'operating_profit')
        forecast_op = TRADE_DATA.get_consensus_forecast(date, 'operating_profit')
        
        sue = (actual_op - forecast_op) / abs(forecast_op)
        return sue.rename('factor_value').to_frame() 