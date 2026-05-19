from __future__ import annotations
import pandas as pd

from collections import defaultdict
from typing import Any
from dataclasses import dataclass , field
from threading import Lock

from src.data import DATAVENDOR
from src.res.factor.calculator import MomentumFactor , CorrelationFactor , VolatilityFactor

from src.func.transform import time_weight , apply_ols

# Guard creation of per-(n_months, date) locks so two threads never get different
# Lock instances for the same cache key (defaultdict is not thread-safe alone).
FF3_fit_lock_keeper : Lock = Lock()
FF3_fit_locks : dict[tuple[int,int],Lock] = defaultdict(Lock)

@dataclass
class FF3_Model:
    stk : pd.DataFrame
    mkt : pd.DataFrame | pd.Series
    hml : pd.DataFrame | pd.Series
    smb : pd.DataFrame | pd.Series
    lock : Lock = field(default_factory=Lock)

    def __post_init__(self):
        ...

    def fit(self):
        ff3_x = self.mkt.merge(self.hml , on = 'date').merge(self.smb , on = 'date')
        ff3_x.columns = ['mkt' , 'hml' , 'smb']
        ff3_x['intercept'] = 1.
        self.ff3_x = ff3_x.loc[:,['intercept' , 'mkt' , 'hml' , 'smb']]

        self.coef = apply_ols(self.ff3_x , self.stk , intercept=False)
        self.pred = self.ff3_x.values @ self.coef
        self.resid = self.stk - self.pred
        self.r2    = 1 - (self.resid ** 2).sum() / (self.stk ** 2).sum()
        return self

    def resid_mom(self , half_life_ratio = 0.5):
        with self.lock:
            return (self.resid * time_weight(len(self.resid) , int(half_life_ratio * len(self.resid)))[:,None]).sum()

    def alpha(self):
        with self.lock:
            return pd.Series(self.coef[0] , index = self.stk.columns)
    
    def resid_vol(self):
        with self.lock:
            return self.resid.std()
    
    def resid_skew(self):
        with self.lock:
            return self.resid.skew()
    
    def resid_kurt(self):
        with self.lock:
            return self.resid.kurt()
    
class FamaFrench3:
    """
    thread safe Fama-French 3-Factor Model
    """
    _cache_models : dict[tuple[int,int],FF3_Model] = {}
    
    def __init__(self , n_months : int , date : int):
        assert n_months in [1, 2, 3, 6, 12], f'n_months must be in [1, 2, 3, 6, 12] , got {n_months}'
        assert date > 0 , f'date must be greater than 0 , got {date}'
        self.n_months = n_months
        self.date = date
        self.fit()

    @staticmethod
    def group_ret(df : pd.DataFrame) -> pd.Series:
        df_new = df.groupby('date')[['mv_add' , 'mv']].sum()
        return df_new['mv_add'] / df_new['mv']

    @classmethod
    def mkt(cls , df : pd.DataFrame):
        return cls.group_ret(df).rename('mkt')

    @classmethod
    def smb(cls , df : pd.DataFrame) -> pd.Series:
        mv_rank : pd.Series | Any = df['mv'].groupby('date').rank(pct = True)
        ret_B = cls.group_ret(df.loc[mv_rank >= 0.9])
        ret_S = cls.group_ret(df.loc[mv_rank < 0.5])
        return (ret_S - ret_B).rename('smb')
    
    @classmethod
    def hml(cls , df : pd.DataFrame) -> pd.Series:
        bp_rank : pd.Series | Any = df['bp'].groupby('date').rank(pct = True)
        ret_H = cls.group_ret(df.loc[bp_rank >= 2/3])
        ret_L = cls.group_ret(df.loc[bp_rank < 1/3])
        return (ret_H - ret_L).rename('hml')

    def fit(self):
        start , end = DATAVENDOR.CALENDAR.td_start_end(self.date , self.n_months , 'm')

        rets = DATAVENDOR.TRADE.get_returns(start , end , mask = False , pivot = False).rename(columns={'pctchange':'ret'})
        mv   = DATAVENDOR.TRADE.get_mv(start , end , mv_type = 'circ_mv' , pivot = False , prev=True).rename(columns={'circ_mv':'mv'})
        btop = 1 / DATAVENDOR.TRADE.get_val_data(start , end , 'pb' , pivot = False , prev = True).rename(columns={'pb':'bp'})
        rets = rets.merge(mv , on = ['date' , 'secid']).merge(btop , on = ['date' , 'secid'])
        rets['mv_add']= rets['mv'] * rets['ret']
        stk = rets.pivot_table('ret' , 'date' , 'secid')
        mkt = self.mkt(rets).to_frame()
        hml = self.hml(rets)
        smb = self.smb(rets)

        self.model = FF3_Model(stk , mkt , hml , smb).fit()
        return self
    
    @classmethod
    def get_ff3_model(cls , n_months : int , date : int) -> FF3_Model:
        key = (n_months, date)
        # Fast path: model is immutable after fit; concurrent reads need no lock.
        if key in cls._cache_models:
            return cls._cache_models[key]
        with FF3_fit_lock_keeper:
            lock = FF3_fit_locks[key]
        with lock:
            # Must re-check inside lock: Factor.Factor / updater run many ff_* factors
            # on the same date in a thread pool; an outer "not in cache" lets every
            # waiter repeat the expensive fit() unless we guard the dict here.
            if key not in cls._cache_models:
                cls._cache_models[key] = FamaFrench3(n_months, date).model
            return cls._cache_models[key]

class ff_mom_1m(MomentumFactor):
    init_date = 20110101
    description = '1个月ff3残差动量'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(1, date)
        return ff3.resid_mom()
    
class ff_mom_2m(MomentumFactor):
    init_date = 20110101
    description = '2个月ff3残差动量'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(2, date)
        return ff3.resid_mom()
    
class ff_mom_3m(MomentumFactor):
    init_date = 20110101
    description = '3个月ff3残差动量'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(3, date)
        return ff3.resid_mom()
    
class ff_mom_6m(MomentumFactor):
    init_date = 20110101
    description = '6个月ff3残差动量'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(6, date)
        return ff3.resid_mom()
    
class ff_mom_12m(MomentumFactor):
    init_date = 20110101
    description = '12个月ff3残差动量'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(12, date)
        return ff3.resid_mom()
    
class ff_r2_1m(CorrelationFactor):
    init_date = 20110101
    description = '1个月ff3模型R2'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(1, date)
        return ff3.r2
    
class ff_r2_2m(CorrelationFactor):
    init_date = 20110101
    description = '2个月ff3模型R2'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(2, date)
        return ff3.r2
    
class ff_r2_3m(CorrelationFactor):
    init_date = 20110101
    description = '3个月ff3模型R2'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(3, date)
        return ff3.r2
    
class ff_r2_6m(CorrelationFactor):
    init_date = 20110101
    description = '6个月ff3模型R2'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(6, date)
        return ff3.r2
    
class ff_r2_12m(CorrelationFactor):
    init_date = 20110101
    description = '12个月ff3模型R2'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(12, date)
        return ff3.r2
    
class ff_alpha_1m(MomentumFactor):
    init_date = 20110101
    description = '1个月ff3模型alpha'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(1, date)
        return ff3.alpha()
    
class ff_alpha_2m(MomentumFactor):
    init_date = 20110101
    description = '2个月ff3模型alpha'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(2, date)
        return ff3.alpha()
    
class ff_alpha_3m(MomentumFactor):
    init_date = 20110101
    description = '3个月ff3模型alpha'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(3, date)
        return ff3.alpha()
    
class ff_alpha_6m(MomentumFactor):
    init_date = 20110101
    description = '6个月ff3模型alpha'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(6, date)
        return ff3.alpha()
    
class ff_alpha_12m(MomentumFactor):
    init_date = 20110101
    description = '12个月ff3模型alpha'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(12, date)
        return ff3.alpha()
    
class ff_resvol_1m(VolatilityFactor):
    init_date = 20110101
    description = '1个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(1, date)
        return ff3.resid_vol()
    
class ff_resvol_2m(VolatilityFactor):
    init_date = 20110101
    description = '2个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(2, date)
        return ff3.resid_vol()
    
class ff_resvol_3m(VolatilityFactor):
    init_date = 20110101
    description = '3个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(3, date)
        return ff3.resid_vol()
    
class ff_resvol_6m(VolatilityFactor):
    init_date = 20110101
    description = '6个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(6, date)
        return ff3.resid_vol()
    
class ff_resvol_12m(VolatilityFactor):
    init_date = 20110101
    description = '12个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(12, date)
        return ff3.resid_vol()

class ff_resskew_1m(VolatilityFactor):
    init_date = 20110101
    description = '1个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(1, date)
        return ff3.resid_skew()

class ff_resskew_2m(VolatilityFactor):
    init_date = 20110101
    description = '2个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(2, date)
        return ff3.resid_skew()
    
class ff_resskew_3m(VolatilityFactor):
    init_date = 20110101
    description = '3个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(3, date)
        return ff3.resid_skew()
    
class ff_resskew_6m(VolatilityFactor):
    init_date = 20110101
    description = '6个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(6, date)
        return ff3.resid_skew()
    
class ff_resskew_12m(VolatilityFactor):
    init_date = 20110101
    description = '12个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(12, date)
        return ff3.resid_skew()
    
class ff_reskurt_1m(VolatilityFactor):
    init_date = 20110101
    description = '1个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(1, date)
        return ff3.resid_kurt()
    
class ff_reskurt_2m(VolatilityFactor):
    init_date = 20110101
    description = '2个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(2, date)
        return ff3.resid_kurt()
    
class ff_reskurt_3m(VolatilityFactor):
    init_date = 20110101
    description = '3个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(3, date)
        return ff3.resid_kurt()
    
class ff_reskurt_6m(VolatilityFactor):
    init_date = 20110101
    description = '6个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(6, date)
        return ff3.resid_kurt()
    
class ff_reskurt_12m(VolatilityFactor):
    init_date = 20110101
    description = '12个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        ff3 = FamaFrench3.get_ff3_model(12, date)
        return ff3.resid_kurt()
