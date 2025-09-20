import pandas as pd

from typing import Any
from dataclasses import dataclass

from src.data import DATAVENDOR
from src.res.factor.calculator import MomentumFactor , CorrelationFactor , VolatilityFactor

from src.func.transform import time_weight , apply_ols
from src.func.singleton import singleton

@dataclass
class FF3_Model:
    stk : pd.DataFrame
    mkt : pd.DataFrame | pd.Series
    hml : pd.DataFrame | pd.Series
    smb : pd.DataFrame | pd.Series

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
    
class FamaFrench3:
    N_MONTHS : int = -1
    def __init__(self , date):
        assert self.N_MONTHS > 0 , self.N_MONTHS
        if getattr(self, 'date' , None) is None or self.date != int(date):
            self.date = int(date)
            self.fit(self.date)

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

    def fit(self , date , half_life = 0 , min_finite_ratio = 0.25):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , self.N_MONTHS , 'm')

        rets = DATAVENDOR.TRADE.get_returns(start_date , end_date , mask = False , pivot = False).rename(columns={'pctchange':'ret'})
        mv   = DATAVENDOR.TRADE.get_mv(start_date , end_date , mv_type = 'circ_mv' , pivot = False , prev=True).rename(columns={'circ_mv':'mv'})
        btop = 1 / DATAVENDOR.TRADE.get_val_data(start_date , end_date , 'pb' , pivot = False , prev = True).rename(columns={'pb':'bp'})
        rets = rets.merge(mv , on = ['date' , 'secid']).merge(btop , on = ['date' , 'secid'])
        rets['mv_add']= rets['mv'] * rets['ret']
        stk = rets.pivot_table('ret' , 'date' , 'secid')
        mkt = self.mkt(rets).to_frame()
        hml = self.hml(rets)
        smb = self.smb(rets)

        self.model = FF3_Model(stk , mkt , hml , smb).fit()
        return self
    
    @staticmethod
    def select_ff3(n_months : int):
        if n_months == 1:
            return FF3_1m
        elif n_months == 2:
            return FF3_2m
        elif n_months == 3:
            return FF3_3m
        elif n_months == 6:
            return FF3_6m
        elif n_months == 12:
            return FF3_12m
        else:
            raise ValueError(f'n_months must be in [1, 2, 3, 6, 12] , got {n_months}')
        
    def resid_mom(self , half_life_ratio = 0.5):
        return (self.model.resid * time_weight(len(self.model.resid) , int(half_life_ratio * len(self.model.resid)))[:,None]).sum()
    
    def r2(self):
        return self.model.r2
    
    def alpha(self):
        return pd.Series(self.model.coef[0] , index = self.model.stk.columns)
    
    def resid_vol(self):
        return self.model.resid.std()
    
    def resid_skew(self):
        return self.model.resid.skew()
    
    def resid_kurt(self):
        return self.model.resid.kurt()

@singleton    
class FF3_1m(FamaFrench3):
    N_MONTHS = 1
@singleton
class FF3_2m(FamaFrench3):
    N_MONTHS = 2
@singleton
class FF3_3m(FamaFrench3):
    N_MONTHS = 3    
@singleton
class FF3_6m(FamaFrench3):
    N_MONTHS = 6
@singleton
class FF3_12m(FamaFrench3):
    N_MONTHS = 12

def select_ff3(n_months : int):
    if n_months == 1:
        return FF3_1m
    elif n_months == 2:
        return FF3_2m
    elif n_months == 3:
        return FF3_3m
    elif n_months == 6:
        return FF3_6m
    elif n_months == 12:
        return FF3_12m
    else:
        raise ValueError(f'n_months must be in [1, 2, 3, 6, 12] , got {n_months}')

class ff_mom_1m(MomentumFactor):
    init_date = 20110101
    description = '1个月ff3残差动量'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(1)(date).resid_mom()
    
class ff_mom_2m(MomentumFactor):
    init_date = 20110101
    description = '2个月ff3残差动量'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(2)(date).resid_mom()
    
class ff_mom_3m(MomentumFactor):
    init_date = 20110101
    description = '3个月ff3残差动量'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(3)(date).resid_mom()
    
class ff_mom_6m(MomentumFactor):
    init_date = 20110101
    description = '6个月ff3残差动量'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(6)(date).resid_mom()
    
class ff_mom_12m(MomentumFactor):
    init_date = 20110101
    description = '12个月ff3残差动量'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(12)(date).resid_mom()
    
class ff_r2_1m(CorrelationFactor):
    init_date = 20110101
    description = '1个月ff3模型R2'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(1)(date).r2()
    
class ff_r2_2m(CorrelationFactor):
    init_date = 20110101
    description = '2个月ff3模型R2'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(2)(date).r2()
    
class ff_r2_3m(CorrelationFactor):
    init_date = 20110101
    description = '3个月ff3模型R2'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(3)(date).r2()
    
class ff_r2_6m(CorrelationFactor):
    init_date = 20110101
    description = '6个月ff3模型R2'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(6)(date).r2()
    
class ff_r2_12m(CorrelationFactor):
    init_date = 20110101
    description = '12个月ff3模型R2'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(12)(date).r2()    
    
class ff_alpha_1m(MomentumFactor):
    init_date = 20110101
    description = '1个月ff3模型alpha'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(1)(date).alpha()
    
class ff_alpha_2m(MomentumFactor):
    init_date = 20110101
    description = '2个月ff3模型alpha'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(2)(date).alpha()
    
class ff_alpha_3m(MomentumFactor):
    init_date = 20110101
    description = '3个月ff3模型alpha'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(3)(date).alpha()
    
class ff_alpha_6m(MomentumFactor):
    init_date = 20110101
    description = '6个月ff3模型alpha'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(6)(date).alpha()
    
class ff_alpha_12m(MomentumFactor):
    init_date = 20110101
    description = '12个月ff3模型alpha'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(12)(date).alpha()
    
class ff_resvol_1m(VolatilityFactor):
    init_date = 20110101
    description = '1个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(1)(date).resid_vol()
    
class ff_resvol_2m(VolatilityFactor):
    init_date = 20110101
    description = '2个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(2)(date).resid_vol()
    
class ff_resvol_3m(VolatilityFactor):
    init_date = 20110101
    description = '3个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(3)(date).resid_vol()
    
class ff_resvol_6m(VolatilityFactor):
    init_date = 20110101
    description = '6个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(6)(date).resid_vol()
    
class ff_resvol_12m(VolatilityFactor):
    init_date = 20110101
    description = '12个月ff3模型残差波动率'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(12)(date).resid_vol()

class ff_resskew_1m(VolatilityFactor):
    init_date = 20110101
    description = '1个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(1)(date).resid_skew()

class ff_resskew_2m(VolatilityFactor):
    init_date = 20110101
    description = '2个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(2)(date).resid_skew()
    
class ff_resskew_3m(VolatilityFactor):
    init_date = 20110101
    description = '3个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(3)(date).resid_skew()
    
class ff_resskew_6m(VolatilityFactor):
    init_date = 20110101
    description = '6个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(6)(date).resid_skew()
    
class ff_resskew_12m(VolatilityFactor):
    init_date = 20110101
    description = '12个月ff3模型残差偏度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(12)(date).resid_skew()
    
class ff_reskurt_1m(VolatilityFactor):
    init_date = 20110101
    description = '1个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(1)(date).resid_kurt()
    
class ff_reskurt_2m(VolatilityFactor):
    init_date = 20110101
    description = '2个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(2)(date).resid_kurt()
    
class ff_reskurt_3m(VolatilityFactor):
    init_date = 20110101
    description = '3个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(3)(date).resid_kurt()
    
class ff_reskurt_6m(VolatilityFactor):
    init_date = 20110101
    description = '6个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(6)(date).resid_kurt()
    
class ff_reskurt_12m(VolatilityFactor):
    init_date = 20110101
    description = '12个月ff3模型残差峰度'

    def calc_factor(self , date : int):
        return FamaFrench3.select_ff3(12)(date).resid_kurt()
