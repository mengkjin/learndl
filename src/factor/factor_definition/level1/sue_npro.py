import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class sue_npro(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '未预期-净利润'
    
    def calc_factor(self, date: int):
        """计算净利润的未预期因子
        1. 获取实际净利润和预测净利润数据
        2. 计算未预期值
        """
        # TODO: 需要定义获取财务数据的接口
        actual_npro = TRADE_DATA.get_income_statement(date, 'net_profit')
        forecast_npro = TRADE_DATA.get_consensus_forecast(date, 'net_profit')
        
        sue = (actual_npro - forecast_npro) / abs(forecast_npro)
        return sue.rename('factor_value').to_frame() 