import numpy as np

from typing import Literal

from src.data import DATAVENDOR
from src.res.factor.calculator import AdjustmentFactor

__all__ = [
    'rec_npro_12m' , 'rec_npro_6m' , 'rec_npro_3m' , 'rec_npro_6m_anndt' ,
    'upnum_npro_12m' , 'upnum_npro_6m' , 'upnum_npro_3m' , 'upnum_npro_6m_anndt' ,
    'uppct_npro_12m' , 'uppct_npro_6m' , 'uppct_npro_3m' , 'uppct_npro_6m_anndt'
]

def get_npro_adjustment(date : int , n_month : int , type : Literal['rec' , 'upnum' , 'uppct'] , 
                        within_ann_days : int | None = None , ):
    target_quarter = f'{date // 10000}Q4' # noqa
    start_date = DATAVENDOR.CALENDAR.cd(date , -30 * n_month) # noqa

    df = DATAVENDOR.ANALYST.get_trailing_reports(date , n_month + 6).set_index(['secid','org_name','report_date'])
    df = df.query('quarter == @target_quarter').sort_index().groupby(['secid','org_name'])['np'].\
        pct_change(fill_method = None).dropna().reset_index()
    df = df.query('report_date >= @start_date').set_index(['secid','report_date'])

    if within_ann_days is not None:
        assert within_ann_days > 0 , 'within_ann_days must be positive'
        ann_cal = DATAVENDOR.IS.get_ann_calendar(date , after_days = within_ann_days , within_days = n_month * 30 + 5).\
            rename_axis(index = {'date':'report_date'}).reindex(df.index)
        df = df[ann_cal['anndt'] > 0]

    df = df.reset_index().groupby(['secid','org_name']).last()
    if type == 'rec':
        df = DATAVENDOR.ANALYST.weighted_val(df , date , 'np')
    elif type == 'upnum':
        df['np'] = df['np'] > 0
        df = df.groupby(['secid'])['np'].sum()
    elif type == 'uppct':
        df['np'] = np.sign(df['np'])
        df = df.groupby(['secid'])['np'].mean()
    return df    

class rec_npro_12m(AdjustmentFactor):
    init_date = 20110101
    description = '12个月盈利预测上调幅度'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 12 , 'rec')
    
class rec_npro_6m(AdjustmentFactor):
    init_date = 20110101
    description = '6个月盈利预测上调幅度'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 6 , 'rec')
    
class rec_npro_3m(AdjustmentFactor):
    init_date = 20110101
    description = '3个月盈利预测上调幅度'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 3 , 'rec')
    
class rec_npro_6m_anndt(AdjustmentFactor):
    init_date = 20110101
    description = '6个月盈利预测上调幅度,公告日后7天'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 6 , 'rec' , within_ann_days = 7)

class upnum_npro_12m(AdjustmentFactor):
    init_date = 20110101
    description = '12个月分析师盈利上修数量'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 12 , 'upnum')  

class upnum_npro_6m(AdjustmentFactor):
    init_date = 20110101
    description = '6个月分析师盈利上修数量'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 6 , 'upnum')  

class upnum_npro_3m(AdjustmentFactor):
    init_date = 20110101
    description = '3个月分析师盈利上修数量'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 3 , 'upnum')
    
class upnum_npro_6m_anndt(AdjustmentFactor):
    init_date = 20110101
    description = '6个月分析师盈利上修数量,公告日后7天'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 6 , 'upnum' , within_ann_days = 7)

class uppct_npro_12m(AdjustmentFactor):
    init_date = 20110101
    description = '12个月分析师盈利上修占比'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 12 , 'uppct')
    
class uppct_npro_6m(AdjustmentFactor):
    init_date = 20110101
    description = '6个月分析师盈利上修占比'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 6 , 'uppct')
    
class uppct_npro_3m(AdjustmentFactor):
    init_date = 20110101
    description = '3个月分析师盈利上修占比'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 3 , 'uppct')

class uppct_npro_6m_anndt(AdjustmentFactor):
    init_date = 20110101
    description = '6个月分析师盈利上修占比,公告日后7天'
    
    def calc_factor(self, date: int):
        return get_npro_adjustment(date , 6 , 'uppct' , within_ann_days = 7)