import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class cov_inst_3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '近三月跟踪股票机构数量'
    
    def calc_factor(self, date: int):
        """计算近3个月跟踪机构数量
        1. 获取机构覆盖数据
        2. 统计3个月内发布报告的机构数量
        """
        # TODO: 需要定义获取分析师覆盖数据的接口
        start_date = TradeDate(date) - 60
        inst_coverage = TRADE_DATA.get_analyst_coverage(start_date, date)
        
        inst_count = inst_coverage.groupby('secid').size()
        return inst_count.rename('factor_value').to_frame() 