import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class cov_inst_anndt(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '近期跟踪股票机构数量'
    
    def calc_factor(self, date: int):
        """计算近期跟踪机构数量
        1. 获取最近一次财报发布日期
        2. 统计发布报告的机构数量
        """
        # TODO: 需要定义获取分析师覆盖数据的接口
        last_report_date = TRADE_DATA.get_last_report_date(date)
        inst_coverage = TRADE_DATA.get_analyst_coverage(last_report_date, date)
        
        inst_count = inst_coverage.groupby('secid').size()
        return inst_count.rename('factor_value').to_frame() 