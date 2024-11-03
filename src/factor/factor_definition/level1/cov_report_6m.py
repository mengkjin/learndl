import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class cov_report_6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '近六月跟踪股票报告数量'
    
    def calc_factor(self, date: int):
        """计算近6个月研报数量
        1. 获取研报发布数据
        2. 统计6个月内的研报数量
        """
        # TODO: 需要定义获取研报数据的接口
        start_date = TradeDate(date) - 120
        reports = TRADE_DATA.get_analyst_reports(start_date, date)
        
        report_count = reports.groupby('secid').size()
        return report_count.rename('factor_value').to_frame() 