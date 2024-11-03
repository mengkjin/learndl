import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class cash_ta_trend(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'liquidity'
    description = 'cash_ta趋势'
    
    def calc_factor(self, date: int):
        """计算现金资产比的趋势
        1. 获取最近4个季度的现金及现金等价物和总资产数据
        2. 计算比值的趋势
        """
        # TODO: 需要定义获取财务数据的接口
        quarters = range(4)
        cash_data = pd.DataFrame()
        ta_data = pd.DataFrame()
        
        for i in quarters:
            offset_date = TradeDate(date) - i * 60  # 约一个季度的交易日
            cash = TRADE_DATA.get_balance_sheet(offset_date, 'cash_equivalents')
            ta = TRADE_DATA.get_balance_sheet(offset_date, 'total_assets')
            cash_data[f'q{i}'] = cash
            ta_data[f'q{i}'] = ta
            
        ratios = cash_data / ta_data
        # 计算趋势(使用简单线性回归斜率)
        trend = ratios.apply(lambda x: np.polyfit(quarters, x, 1)[0], axis=1)
        
        return trend.rename('factor_value').to_frame() 