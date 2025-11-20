from typing import Literal

from src.basic import CALENDAR
from src.data import DATAVENDOR
from src.res.factor.calculator import MomentumFactor    

def get_profit_ann_dt(date : int):
    ann_dt = DATAVENDOR.IS.get_ann_dt(date , 1 , 180)
    ann_dt = ann_dt['td_forward'].reset_index('end_date' , drop=True).rename('date')
    positive_yoy = DATAVENDOR.get_fin_yoy('npro@qtr' , date , 4).dropna().groupby('secid').last().iloc[:,0]
    ann_dt = ann_dt[positive_yoy.reindex(ann_dt.index) > 0]
    return ann_dt.reset_index(drop=False)

def get_pead_df(date : int , price_type : Literal['open' , 'low'] , rank_pct : bool = True , running_days : int = 20):
    ann_cal = get_profit_ann_dt(date).set_index(['date','secid'])
    dates = ann_cal.index.get_level_values('date')

    end_date = dates.max()
    ann_cal['0'] = dates
    for i in range(running_days):
        dates = CALENDAR.td_array(dates , -1)
        ann_cal[f'-{i+1}'] = dates
    start_date = CALENDAR.td(dates.min() , -20)
    ann_cal = ann_cal.reset_index('date' , drop = True).\
        melt(var_name = 'prev_day' , value_name = 'date' , ignore_index = False).\
        reset_index(drop = False).set_index(['date' , 'secid'])
    ann_cal['prev_day'] = ann_cal['prev_day'].astype(int)

    quotes = DATAVENDOR.TRADE.get_quotes(start_date , end_date , ['preclose' , price_type])
    mv = DATAVENDOR.TRADE.get_mv(start_date , end_date , 'circ_mv')

    quotes['circ_mv'] = mv['circ_mv']
    quotes['pct_change'] = quotes[price_type] / quotes['preclose'] - 1

    quotes['market_pct_change'] = quotes.groupby('date').apply(lambda x: (x['pct_change'] * x['circ_mv']).sum() / x['circ_mv'].sum()).\
        loc[quotes.index.get_level_values('date')].values
    quotes['act_pct_change'] = quotes['pct_change'] - quotes['market_pct_change']
    if rank_pct:
        quotes['act_pct_change'] = quotes.groupby('date')['act_pct_change'].rank(pct=True)
    
    pred_df = ann_cal.assign(act_pct_change = quotes['act_pct_change'].reindex(ann_cal.index))
    pred_df = pred_df.reset_index().pivot_table(index = ['secid'] , columns = 'prev_day' , values = 'act_pct_change')

    return pred_df

class _pead_calculator:
    """
    to calculate pead factors faster
    """
    running_days = 20
    _instance = None

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , date : int):
        assert date > 0 , f'date must be positive integer , got {date}'
        if self.date != date:
            self.date = date
            self.calc_ann_cal()
            self.calc_pead_quotes()
    
    @property
    def date(self):
        if not hasattr(self , '_date'):
            return -1
        else:
            return self._date

    @date.setter
    def date(self , date : int):
        self._date = date

    def calc_ann_cal(self):
        ann_cal = get_profit_ann_dt(self.date).set_index(['date','secid'])
        dates = ann_cal.index.get_level_values('date')

        end_date = dates.max()
        ann_cal['0'] = dates
        for i in range(self.running_days):
            dates = CALENDAR.td_array(dates , -1)
            ann_cal[f'-{i+1}'] = dates
        start_date = CALENDAR.td(dates.min() , -20)
        ann_cal = ann_cal.reset_index('date' , drop = True).\
            melt(var_name = 'prev_day' , value_name = 'date' , ignore_index = False).\
            reset_index(drop = False).set_index(['date' , 'secid'])
        ann_cal['prev_day'] = ann_cal['prev_day'].astype(int)
        self.ann_cal = ann_cal
        self.start_date = start_date
        self.end_date = end_date

    def calc_pead_quotes(self):
        quotes = DATAVENDOR.TRADE.get_quotes(self.start_date , self.end_date , ['preclose' , 'open' , 'low'])
        mv = DATAVENDOR.TRADE.get_mv(self.start_date , self.end_date , 'circ_mv')

        quotes['circ_mv'] = mv['circ_mv']
        quotes['open_rtn'] = quotes['open'] / quotes['preclose'] - 1
        quotes['low_rtn'] = quotes['low'] / quotes['preclose'] - 1

        quotes['market_open_rtn'] = quotes.groupby('date').apply(lambda x: (x['open_rtn'] * x['circ_mv']).sum() / x['circ_mv'].sum()).\
            loc[quotes.index.get_level_values('date')].values
        quotes['market_low_rtn'] = quotes.groupby('date').apply(lambda x: (x['low_rtn'] * x['circ_mv']).sum() / x['circ_mv'].sum()).\
            loc[quotes.index.get_level_values('date')].values
        quotes['act_open_rtn'] = quotes['open_rtn'] - quotes['market_open_rtn']
        quotes['act_low_rtn'] = quotes['low_rtn'] - quotes['market_low_rtn']
        quotes['act_open_rtn_rank'] = quotes.groupby('date')['act_open_rtn'].rank(pct=True)
        quotes['act_low_rtn_rank'] = quotes.groupby('date')['act_low_rtn'].rank(pct=True)
        self.quotes = quotes.loc[:,['act_open_rtn' , 'act_low_rtn' , 'act_open_rtn_rank' , 'act_low_rtn_rank']]

    def pead_df(self , price_type : Literal['open' , 'low'] , rank_pct : bool = True):
        rtn_str = f'act_{price_type}_rtn'
        if rank_pct:
            rtn_str += '_rank'
        pred_df = self.ann_cal.assign(rtn = self.quotes[rtn_str].reindex(self.ann_cal.index))
        pred_df = pred_df.reset_index().pivot_table(index = ['secid'] , columns = 'prev_day' , values = 'rtn')
        return pred_df


class pead_aog(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日开盘跳空超额'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = _pead_calculator(date).pead_df('open' , rank_pct = False)
        return pead_df.iloc[:,-1]
    
class pead_alg(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日最低价超额'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = _pead_calculator(date).pead_df('low' , rank_pct = False)
        return pead_df.iloc[:,-1]

class pead_aog_rank(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日开盘跳空超额当日排名分位'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = _pead_calculator(date).pead_df('open' , rank_pct = True)
        return pead_df.iloc[:,-1]

class pead_alg_rank(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日最低价超额当日排名分位'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = _pead_calculator(date).pead_df('low' , rank_pct = True)
        return pead_df.iloc[:,-1]

class pead_aog_rank_demax(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日开盘跳空超额当日排名分位剔除前20日排名最大值'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = _pead_calculator(date).pead_df('open')
        return pead_df.iloc[:,-1] - pead_df.iloc[:,:-1].max(axis = 1)

class pead_alg_rank_demax(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日前20日最低价超额排名分位剔除前20日排名最大值'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = _pead_calculator(date).pead_df('low')
        return pead_df.iloc[:,-1] - pead_df.iloc[:,:-1].max(axis = 1)

class pead_aog_rank_quantile(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日前20日开盘跳空超额排名20%分位'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = _pead_calculator(date).pead_df('open')
        return pead_df.iloc[:,:-1].quantile(0.2 , axis = 1)

class pead_alg_rank_quantile(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日前20日最低价超额排名20%分位'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = _pead_calculator(date).pead_df('low')
        return pead_df.iloc[:,:-1].quantile(0.2 , axis = 1)