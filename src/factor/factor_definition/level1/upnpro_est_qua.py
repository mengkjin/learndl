import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class upnpro_est_qua(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'surprise'
    description = '季度净利润超预期幅度'
    
    def calc_factor(self, date: int):
        """计算季度净利润超预期幅度
        1. 获取分析师预测的季度净利润
        2. 获取实际季度净利润
        3. 计算超预期幅度
        """
        # TODO: 需要定义获取分析师预测数据的接口
        est_npro = TRADE_DATA.get_analyst_est_npro_quarterly(date)
        actual_npro = TRADE_DATA.get_npro_quarterly(date)
        
        surprise = (actual_npro - est_npro) / abs(est_npro)
        return surprise.rename('factor_value').to_frame() 