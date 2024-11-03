import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class roa_q(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '季度总资产收益率'
    
    def calc_factor(self, date: int):
        """计算季度总资产收益率
        1. 获取季度净利润和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        net_profit = TRADE_DATA.get_income_statement(date, 'net_profit', 'quarterly')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = net_profit / total_assets
        return ratio.rename('factor_value').to_frame() 