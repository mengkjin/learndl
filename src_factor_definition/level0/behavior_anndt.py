import numpy as np
import pandas as pd

from typing import Literal
from src.factor.classes import StockFactorCalculator
from src.data import TSData

class anndt_phigh(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '公告日最高价距离'
    
    def calc_factor(self, date: int):
        ann_dt = TSData.FINA.get_ann_dt(date , 1 , within_days=365).rename(columns={'td_forward':'date'})
        start_date , end_date = ann_dt['date'].min() , ann_dt['date'].max()
        
        quotes = TSData.TRADE.get_quotes(start_date , end_date , ['close' , 'high'] , pivot = False)
        ann_dt_perf = ann_dt.merge(quotes , on = ['secid','date'])
        ann_dt_perf['phigh'] = ann_dt_perf['close'] / ann_dt_perf['high'] - 1
        return ann_dt_perf['phigh']
    
class mom_aog(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '公告日次日超额收益'
    
    def calc_factor(self, date: int):
        ann_dt = TSData.FINA.get_ann_dt(date , 1 , within_days=365)
        ann_dt['date'] = TSData.CALENDAR.offset(ann_dt['td'] , 1)

        start_date , end_date = ann_dt['date'].min() , ann_dt['date'].max()
        
        quotes = TSData.MODEL.get_exret(start_date , end_date , pivot = False)
        ann_dt_perf = ann_dt.merge(quotes , on = ['secid','date'])
        return ann_dt_perf['resid']


class mom_aaa(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '公告日后3日超额收益'
    
    def calc_factor(self, date: int):
        ann_dt = TSData.FINA.get_ann_dt(date , 1 , within_days=365)
        ann_dt['date0'] = TSData.CALENDAR.offset(ann_dt['td'] , 1)
        ann_dt['date1'] = TSData.CALENDAR.offset(ann_dt['td'] , 2)
        ann_dt['date2'] = TSData.CALENDAR.offset(ann_dt['td'] , 3)

        start_date , end_date = ann_dt['date0'].min() , ann_dt['date1'].max()
        quotes = TSData.MODEL.get_exret(start_date , end_date , pivot = False)
        ann_dt = ann_dt.reset_index().melt(
            id_vars=['secid'], value_vars=['date0', 'date1', 'date2'], var_name='date_type', value_name='date')

        ann_dt_perf = ann_dt.merge(quotes , on = ['secid','date'])
        return ann_dt_perf.groupby('secid')['resid'].sum()
