import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class cov_report_anndt(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '近期跟踪股票报告数量'
    
    def calc_factor(self, date: int):
        """计算近期研报数量
        1. 获取最近一次财报发布日期
        2. 统计期间内的研报数量
        """
        # TODO: 需要定义获取研报数据的接口
        last_report_date = TRADE_DATA.get_last_report_date(date)
        reports = TRADE_DATA.get_analyst_reports(last_report_date, date)
        
        report_count = reports.groupby('secid').size()
        return report_count.rename('factor_value').to_frame() 