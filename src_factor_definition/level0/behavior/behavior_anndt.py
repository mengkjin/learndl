import numpy as np
import pandas as pd

from typing import Literal
from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

def date_min_max(date_col : pd.Series):
    return date_col.min() , date_col.max()

class anndt_phigh(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '公告日最高价距离'
    
    def calc_factor(self, date: int):
        ann_dt = DATAVENDOR.IS.get_ann_dt(date , 1 , within_days=365)
        ann_dt['date'] = ann_dt['td_forward']
        start_date , end_date = date_min_max(ann_dt['date'])
        
        quotes = DATAVENDOR.TRADE.get_quotes(start_date , end_date , ['close' , 'high'] , pivot = False)
        ann_dt_perf = ann_dt.merge(quotes , on = ['secid','date'])
        ann_dt_perf['phigh'] = ann_dt_perf['close'] / ann_dt_perf['high'] - 1
        return ann_dt_perf['phigh']
    
class mom_aog(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '公告日次日超额收益'
    
    def calc_factor(self, date: int):
        ann_dt = DATAVENDOR.IS.get_ann_dt(date , 1 , within_days=365)
        ann_dt['date'] = DATAVENDOR.CALENDAR.td_array(ann_dt['td_backward'] , 1)
        start_date , end_date = date_min_max(ann_dt['date'])
        
        quotes = DATAVENDOR.RISK.get_exret(start_date , end_date , pivot = False)
        ann_dt_perf = ann_dt.merge(quotes , on = ['secid','date'])
        return ann_dt_perf['resid']


class mom_aaa(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '公告日后3日超额收益'
    
    def calc_factor(self, date: int):
        ann_dt = DATAVENDOR.IS.get_ann_dt(date , 1 , within_days=365)
        ann_dt['d0'] = DATAVENDOR.CALENDAR.td_array(ann_dt['td_backward'] , 1)
        ann_dt['d1'] = DATAVENDOR.CALENDAR.td_array(ann_dt['td_backward'] , 2)
        ann_dt['d2'] = DATAVENDOR.CALENDAR.td_array(ann_dt['td_backward'] , 3)

        start_date , end_date = ann_dt['d0'].min() , ann_dt['d2'].max()

        quotes = DATAVENDOR.RISK.get_exret(start_date , end_date , pivot = False)
        ann_dt = ann_dt.reset_index().melt(
            id_vars=['secid'], value_vars=['d0', 'd1', 'd2'], var_name='date_type', value_name='new_date').\
            rename(columns={'new_date':'date'})

        ann_dt_perf = ann_dt.merge(quotes , on = ['secid','date'])
        return ann_dt_perf.groupby('secid')['resid'].sum()
