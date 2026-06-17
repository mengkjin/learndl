"""
Momentum PEAD factors for stock level0
"""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Literal , TypeAlias

from src.proj import CALENDAR , Base
from src.data import DATAVENDOR
from src.res.factor.calculator import MomentumFactor

__all__ = [
    'pead_aog' , 'pead_alg' , 'pead_aog_rank' , 'pead_alg_rank' , 
    'pead_aog_rank_demax' , 'pead_alg_rank_demax' , 'pead_aog_rank_quantile' , 
    'pead_alg_rank_quantile'
]

# Guard creation of per-date locks so two threads never get different Lock
# instances for the same cache key (defaultdict is not thread-safe alone).
_pead_lock_keeper : Lock = Lock()
_pead_locks : dict[int , Lock] = defaultdict(Lock)

PeadPriceType : TypeAlias = Literal['open' , 'low']


def get_profit_ann_dt(date : int):
    ann_dt = DATAVENDOR.IS.get_ann_dt(date , 1 , 180)
    ann_dt = ann_dt['td_forward'].reset_index('end_date' , drop=True).rename('date')
    positive_yoy = DATAVENDOR.get_fin_yoy('npro@qtr' , date , 4).dropna().groupby('secid').last().iloc[:,0]
    ann_dt = ann_dt[positive_yoy.reindex(ann_dt.index) > 0]
    return ann_dt.reset_index(drop=False)

class PeadCalculator(Base.BoundLogger):
    """
    Thread-safe, per-date cache for PEAD intermediate data (ann calendar + quotes).
    """
    running_days = 20
    _cache : dict[int , PeadCalculator] = {}

    def __init__(self , date : int , **kwargs):
        assert date > 0 , f'date must be positive integer , got {date}'
        super().__init__(**kwargs)
        self.date = date
        self.lock = Lock()
        self.calc_ann_cal()
        self.calc_pead_quotes()

    @classmethod
    def get(cls , date : int) -> PeadCalculator:
        # Fast path: ann_cal / quotes are fixed after build; concurrent reads need no lock.
        if date in cls._cache:
            return cls._cache[date]
        with _pead_lock_keeper:
            lock = _pead_locks[date]
        with lock:
            # Re-check inside lock: many pead_* factors share the same date in a
            # thread pool; an outer "not in cache" would let every waiter repeat
            # calc_ann_cal / calc_pead_quotes unless we guard the dict here.
            if date not in cls._cache:
                cls._cache[date] = cls(date)
            return cls._cache[date]

    def calc_ann_cal(self):
        ann_cal = get_profit_ann_dt(self.date).set_index(['date','secid'])
        dates = ann_cal.index.get_level_values('date').to_numpy()

        end = dates.max()
        ann_cal['0'] = dates
        for i in range(self.running_days):
            dates = CALENDAR.offset(dates , offset = -1 , type = 'td')
            ann_cal[f'-{i+1}'] = dates
        start = CALENDAR.td(dates.min() , -20)
        ann_cal = ann_cal.reset_index('date' , drop = True).\
            melt(var_name = 'prev_day' , value_name = 'date' , ignore_index = False).\
            reset_index(drop = False).set_index(['date' , 'secid']).astype({'prev_day':int})
        self.ann_cal = ann_cal
        self.start = start
        self.end = end

    def calc_pead_quotes(self):
        quotes = DATAVENDOR.TRADE.get_quotes(self.start , self.end , ['preclose' , 'open' , 'low'])
        mv = DATAVENDOR.TRADE.get_mv(self.start , self.end , 'circ_mv')

        if not quotes.index.is_unique:
            self.logger.error(quotes.index[quotes.index.duplicated()])
            raise ValueError('for PEAD, quotes index must be unique, got stop here')
        if not mv.index.is_unique:
            self.logger.error(mv.index[mv.index.duplicated()])
            raise ValueError('for PEAD, mv index must be unique, got stop here')

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

    def pead_df(self , price_type : PeadPriceType , rank_pct : bool = True):
        with self.lock:
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
        pead_df = PeadCalculator.get(date).pead_df('open' , rank_pct = False)
        return pead_df.iloc[:,-1]
    
class pead_alg(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日最低价超额'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = PeadCalculator.get(date).pead_df('low' , rank_pct = False)
        return pead_df.iloc[:,-1]

class pead_aog_rank(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日开盘跳空超额当日排名分位'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = PeadCalculator.get(date).pead_df('open' , rank_pct = True)
        return pead_df.iloc[:,-1]

class pead_alg_rank(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日最低价超额当日排名分位'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = PeadCalculator.get(date).pead_df('low' , rank_pct = True)
        return pead_df.iloc[:,-1]

class pead_aog_rank_demax(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日开盘跳空超额当日排名分位剔除前20日排名最大值'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = PeadCalculator.get(date).pead_df('open')
        return pead_df.iloc[:,-1] - pead_df.iloc[:,:-1].max(axis = 1)

class pead_alg_rank_demax(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日前20日最低价超额排名分位剔除前20日排名最大值'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = PeadCalculator.get(date).pead_df('low')
        return pead_df.iloc[:,-1] - pead_df.iloc[:,:-1].max(axis = 1)

class pead_aog_rank_quantile(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日前20日开盘跳空超额排名20%分位'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = PeadCalculator.get(date).pead_df('open')
        return pead_df.iloc[:,:-1].quantile(0.2 , axis = 1)

class pead_alg_rank_quantile(MomentumFactor):
    init_date = 20110101
    description = '盈余公告次日前20日最低价超额排名20%分位'
    preprocess = False

    def calc_factor(self , date : int):
        pead_df = PeadCalculator.get(date).pead_df('low')
        return pead_df.iloc[:,:-1].quantile(0.2 , axis = 1)