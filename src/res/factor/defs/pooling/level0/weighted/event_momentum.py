import pandas as pd
import numpy as np

from typing import Literal

from src.res.factor.util import StockFactor
from src.res.factor.calculator import WeightedPoolingCalculator , StockFactorHierarchy
from src.basic import CALENDAR
from src.data.loader.data_vendor import DATAVENDOR


class EventSignal:
    """
    evaluate group performance of a given market event
    """
    test_mode = False
    events = ['selloff_rebound' , 'platform_breakout' , 'high_level_switch']

    def __init__(self):
        self.loaded = False

    def eval(self):
        event_dfs = []
        for event in self.events:
            if event == 'selloff_rebound':
                df = self._selloff_rebound()
            elif event == 'platform_breakout':
                df = self._platform_breakout()
            elif event == 'high_level_switch':
                df = self._high_level_switch()
            else:
                raise ValueError(f'invalid event: {event}')
            assert np.isin(['start' , 'end' , 'event_date' , 'date'] , df.columns).all() , \
                f'event_df must contain start and end columns'
            if 'event' not in df.columns:
                df['event'] = event
            df = df.loc[:,['event' , 'event_date' , 'start' , 'end' , 'date']]
            event_dfs.append(df)
        self.df = pd.concat([d for d in event_dfs if not d.empty]).reset_index(drop = True)
        self.loaded = True
        return self

    @property
    def event_dates(self):
        return self.df['event_date'].sort_values().unique()

    @property
    def factor_dates(self):
        return self.df['date'].sort_values().unique()

    def relative_dates(self , start : int , end : int , trailing : int = 35) -> np.ndarray:
        """
        get relative dates between event dates and factor dates
        give a trailing window > 10 (event lookback window) + 20 (weight proliferation window)
        """
        dates = np.unique(np.concatenate([self.event_dates , self.factor_dates]))
        return CALENDAR.slice(dates , CALENDAR.td(start , -trailing) , end)

    @classmethod
    def _raw_event_df(cls , event : str) -> pd.DataFrame:
        event_df = StockFactorHierarchy.get_factor(event).Load(None)
        if cls.test_mode:
            if event == 'platform_breakout':
                event_df.loc[event_df['date'] == 20250314 , event] = 1
            elif event == 'high_level_switch':
                event_df.loc[event_df['date'] == 20250411 , event] = 1
        event_df = event_df.query(f'{event} == 1')
        return event_df

    @classmethod
    def _selloff_rebound(cls) -> pd.DataFrame:
        """group performance of selloff rebound event"""
        event_df = cls._raw_event_df('selloff_rebound')
        event_df['event_date'] = event_df['date']
        event_df = event_df.rename(columns = {'rebound_start' : 'start' , 'date' : 'end'})
        event_df['date'] = CALENDAR.td_array(event_df['start'] , -1)
        return event_df

    @classmethod
    def _platform_breakout(cls) -> pd.DataFrame:
        """group performance of platform breakout event"""
        event_df = cls._raw_event_df('platform_breakout')

        event_df['event_date'] = event_df['date']
        event_df['start'] = event_df['date']
        event_df['end'] = event_df['date']
        event_df['date'] = CALENDAR.td_array(event_df['date'] , -1)
        return event_df

    @classmethod
    def _high_level_switch(cls) -> pd.DataFrame:
        """group performance of high level switch event"""
        event_df = cls._raw_event_df('high_level_switch')
        event_df['event_date'] = event_df['date']
        event_df['start'] = event_df['date']
        event_df['end'] = event_df['date']
        event_df['date'] = CALENDAR.td_array(event_df['date'] , -1)
        return event_df

class SignedFactor:
    """
    evaluate group performance of a given factor , with a given direction
    can add '+' or '-' to factor names to indicate the direction of the factor
    """
    def __init__(self , signed_name : str):
        self.signed_name = signed_name
        self.factor_name = signed_name.lstrip('-').lstrip('+')
        self.direction = -1 if signed_name.startswith('-') else 1

        self.loaded = False

    def eval(self , dates : np.ndarray | list[int] | int , indent : int = 1 , vb_level : int = 1):
        if isinstance(dates , int):
            dates = [dates]
        self.factor = StockFactorHierarchy.get_factor(self.factor_name).Factor(dates , indent = indent , vb_level = vb_level)
        self.loaded = True
        return self

    @property
    def date(self):
        if not self.loaded:
            raise ValueError('factor not loaded')
        return self.factor.date

    def eval_grp_perf(self , dates : np.ndarray , event_signal : EventSignal , excess : bool = True):
        # get sub fac df and append miscel ret from rebound_start to event_date
        if not self.loaded or not np.isin(dates , self.date).all():
            self.eval(dates)
        facdf = self.factor.frame().query('date in @event_signal.factor_dates').reset_index()
        if facdf.empty:
            self.grp_perf = pd.DataFrame()
        else:
            facdf = facdf.merge(event_signal.df, on = 'date' , how = 'left')
            facdf = DATAVENDOR.get_miscel_ret(facdf)

            grp_perf = self.factor._eval_group_perf(facdf , self.factor.factor_names , excess = excess , direction = self.direction)
            grp_perf['factor_name'] = self.signed_name

            self.grp_perf = grp_perf.merge(event_signal.df , on = 'date' , how = 'left')
        return self

    def frame(self , date_weights : pd.Series | pd.DataFrame | None = None):
        """
        calculate factor frame , with optional date weights
        """
        if date_weights is None or date_weights.empty:
            df = self.factor.frame()
        else:
            if not self.loaded or not np.isin(date_weights.index , self.date).all():
                self.eval(date_weights.index.to_numpy())
            df = self.factor.frame().query('date in @date_weights.index').copy()

        if self.direction == -1:
            df = df.rename(columns = {self.factor_name : self.signed_name})
            df.loc[:,self.signed_name] *= self.direction

        if date_weights is not None and not date_weights.empty:
            if isinstance(date_weights , pd.DataFrame):
                weights = date_weights.loc[df.index.get_level_values('date') , self.signed_name].to_numpy()
            else:
                weights = date_weights.loc[df.index.get_level_values('date')].to_numpy()
            df.loc[:,self.signed_name] *= weights
        return df

class EventFactorWeight:
    """
    evaluate factor weights for a given market event
    """
    momentum_type : Literal['top' , 'topbot'] = 'top'
    momentum_time_decay : bool = True
    ignore_negative_weight : bool = True

    def __init__(self , event_perf : pd.DataFrame , full_dates : np.ndarray , factor_names : list[str]):
        self.event_perf = event_perf
        self.full_dates = full_dates
        self.factor_names = factor_names

    def eval(self):
        if self.event_perf.empty:
            self.weights = pd.DataFrame(
                index = pd.Index([] , name = 'date') , 
                columns = pd.Index(self.factor_names))
            self.full_weights = pd.DataFrame(
                index = pd.Index(self.full_dates , name = 'date') , 
                columns = pd.Index(self.factor_names) ,
                data = 1 / len(self.factor_names))
            return self

        event_perf = self.event_perf.copy()
        event_date_metric = event_perf.sort_values('group').\
            groupby(['event_date' , 'event' , 'factor_name']).\
            apply(self.event_date_metric , include_groups = False).\
            rename('weight').reset_index()
        if self.momentum_time_decay:
            event_date_weight = self.time_decay_event_weight(event_date_metric)
        else:
            event_date_weight = self.natural_event_weight(event_date_metric)

        if self.ignore_negative_weight:
            event_date_weight.loc[event_date_weight['weight'] <= 0 , 'weight'] = 0

        event_date_weight['weight'] = event_date_weight.groupby('event_date')['weight'].transform(self.scale_weights)
        weights = event_date_weight.rename(columns = {'event_date' : 'date'})

        weight_table = weights.pivot_table(index = 'date' , columns = 'factor_name' , values = 'weight').fillna(0)
        self.weights = weight_table.loc[:,self.factor_names]
        trailing_full_dates = np.concatenate([CALENDAR.td_trailing(self.full_dates[0] , 21)[:-1] , self.full_dates])
        self.full_weights = weight_table.reindex(trailing_full_dates).ffill(limit = 20).fillna(1 / weight_table.shape[1]).reindex(self.full_dates)

        return self

    @classmethod
    def event_date_metric(cls , x : pd.DataFrame , **kwargs) -> float:
        top = x.iloc[-1]['group_ret']
        bot = 0 if cls.momentum_type == 'top' else x.iloc[1]['group_ret']
        return top - bot

    @staticmethod
    def scale_weights(weights : pd.Series , **kwargs) -> pd.Series:
        if weights.sum() <= 0:
            y = (weights * 0 + 1) / len(weights)
        else:
            y = weights / weights.sum()
        return y

    @classmethod
    def time_decay_event_weight(cls , event_date_metric : pd.DataFrame , n_window : int = 10 , half_life : int = 5) -> pd.DataFrame:
        weights = []
        for event_date in event_date_metric['event_date'].unique():
            weights.append(cls.prev_events_weight(event_date_metric , event_date , n_window , half_life))
        weights = pd.concat(weights).reset_index(drop = True)
        return weights.loc[:,['event_date' , 'factor_name' , 'weight']].sort_values('event_date')
    
    @classmethod
    def natural_event_weight(cls , event_date_metric : pd.DataFrame) -> pd.DataFrame:
        weights = event_date_metric.groupby(['event_date' , 'factor_name'])['weight'].sum().rename('weight').reset_index()
        return weights.sort_values('event_date')

    @staticmethod
    def prev_events_weight(event_date_weight : pd.DataFrame , event_date : int , n_window : int = 10 , half_life : int = 5) -> pd.DataFrame:
        prev_dates = pd.DataFrame({'n' : range(n_window)} , index = CALENDAR.td_trailing(event_date , n_window))
        prev_wgt = event_date_weight.query('event_date in @prev_dates.index').copy()
        n_days = (prev_dates.loc[prev_wgt['event_date'] , 'n'] - n_window + 1).to_numpy(dtype = float)
        prev_wgt['weight'] *= 2**(n_days / half_life)
        prev_wgt = prev_wgt.groupby('factor_name').sum().reset_index().assign(event_date = event_date)
        return prev_wgt

class MarketEventMomentumFactorWeight:
    """
    evaluate group performance of a given market event , based on market event momentum algorithm
    can add '+' or '-' to factor names to indicate the direction of the factor
    can calculate factor weights for market event momentum algorithm
    """

    def __init__(self , factor_names : list[str]):
        self.factor_names = factor_names
        self.factors : dict[str , SignedFactor] = {factor : SignedFactor(factor) for factor in factor_names}
        
    @property
    def event_df(self):
        return self.event_signal.df

    @property
    def factor_weight(self):
        return self.event_factor_weight.weights

    @property
    def factor_weight_full(self):
        return self.event_factor_weight.full_weights

    def eval_event_signal(self):
        self.event_signal = EventSignal().eval()
        return self
        
    def eval_event_perf(self , start : int , end : int):
        if not hasattr(self , 'event_signal'):
            self.eval_event_signal()
        relative_dates = self.event_signal.relative_dates(start , end)
        dfs = [sfactor.eval_grp_perf(relative_dates , self.event_signal , excess = True).grp_perf for sfactor in self.factors.values()]
        self.event_perf = pd.concat(dfs).reset_index(drop = True)
        return self

    def eval_factor_weights(self , start : int , end : int):
        """evaluate factor weights for a given date range"""
        if not hasattr(self , 'event_perf'):
            self.eval_event_perf(start , end)
        full_dates = CALENDAR.td_within(start , end)
        self.event_factor_weight = EventFactorWeight(self.event_perf , full_dates , self.factor_names).eval()
        return self

    def weighted_factor(self , dates : np.ndarray , name : str = 'weighted_factor') -> StockFactor:
        if len(dates) == 0:
            return StockFactor()
        self.eval_factor_weights(min(dates) , max(dates))
        factor_weights = self.factor_weight_full.loc[dates]
        factor_dfs = [sfactor.frame(factor_weights) for sfactor in self.factors.values()]
        factor_df = pd.concat(factor_dfs , axis = 1).sum(axis = 1).rename(name).to_frame()
        return StockFactor(factor_df)

class event_factor_momentum_test(WeightedPoolingCalculator):
    """
    evaluate group performance of a given market event , based on market event momentum algorithm
    can add '+' or '-' to factor names to indicate the direction of the factor
    can calculate factor weights for market event momentum algorithm
    """
    init_date = 20241201
    update_step = 1
    description = '基于时点动量的因子轮动(测试因子集合)'
    sub_factors : list[str] = [
        'umr_new_1m' , 'btop' , '-risk_lncap' , 'risk_beta'
    ]
    updatable = False

    def calc_factor(self , date : int) -> pd.DataFrame:
        factor_weight = self.get_pooling_weight(date)
        if not np.isin(self.sub_factors , factor_weight.columns).all():
            self.purge_pooling_weight(confirm = True)
            raise ValueError(f'factor weight is missing for {np.setdiff1d(self.sub_factors , factor_weight.columns)}')
        sfactors = [SignedFactor(factor).eval(date).frame(factor_weight) for factor in self.sub_factors]
        factor_df = pd.concat(sfactors , axis = 1).sum(axis = 1).rename(self.factor_name).reset_index(drop = False)
        factor_df = StockFactor.normalize_df(factor_df).drop(columns = ['date'])
        return factor_df

    def calc_pooling_weight(self , start : int | None = None , end : int | None = None , dates : np.ndarray | None = None , overwrite = False , vb_level : int = 1) -> pd.DataFrame:
        """calculate pooling weight of a given date range"""
        if dates is None:
            dates = CALENDAR.slice(CALENDAR.td_within(start , end) , self.init_date , CALENDAR.updated())
        factor_weight = MarketEventMomentumFactorWeight(self.sub_factors)
        factor_weight.eval_factor_weights(min(dates) , max(dates))
        df = factor_weight.factor_weight_full.loc[dates].reset_index(['date'] ,drop = False).rename_axis(None , axis = 1)
        return df

class event_factor_momentum(WeightedPoolingCalculator):
    """
    evaluate group performance of a given market event , based on market event momentum algorithm
    can add '+' or '-' to factor names to indicate the direction of the factor
    can calculate factor weights for market event momentum algorithm
    """
    init_date = 20170101
    update_step = 1
    description = '基于时点动量的因子轮动(生产因子集合)'
    sub_factors : list[str] = [
        '-risk_lncap' , 'risk_beta' ,   # style
        'btop' , 'etop' , 'stop' , 'etop_est' , 'epg_est' , # valuation
        'sales_yoy' , 'npro_yoy' , 'sue_npro' , 'sue_sales' , 'outperform_npro' , # growth
        'roe_qtr' , 'roa_qtr' , 'roe_yoy' , 'roa_yoy' , # profitability
        'analyst_recognition' , 'uppct_npro_3m' , 'rec_npro_3m' , 'outperform_titlepct' , # analyst adjustment
        'dtop' , # dividend
        'pead_aog_rank' , 'pead_alg_rank' , # PEAD
        '-turn_1m' , '-turn_3m' , # liquidity
        'ff_r2_1m' , 'atr_1m' , # volatility
        '-mom_1m' , '-mom_3m' , # reversal
        'umr_new_1m' , 'umr_new_3m' , 'umr_new_6m' , 'umr_new_12m', # umr_new
    ]
    # updatable = False

    def calc_factor(self , date : int) -> pd.DataFrame:
        factor_weight = self.get_pooling_weight(date)
        if not np.isin(self.sub_factors , factor_weight.columns).all():
            self.purge_pooling_weight(confirm = True)
            raise ValueError(f'factor weight is missing for {np.setdiff1d(self.sub_factors , factor_weight.columns)}')
        sfactors = [SignedFactor(factor).eval(date).frame(factor_weight) for factor in self.sub_factors]
        factor_df = pd.concat(sfactors , axis = 1).sum(axis = 1).rename(self.factor_name).reset_index(drop = False)
        factor_df = StockFactor.normalize_df(factor_df).drop(columns = ['date'])
        return factor_df

    def calc_pooling_weight(self , start : int | None = None , end : int | None = None , dates : np.ndarray | None = None , overwrite = False , vb_level : int = 1) -> pd.DataFrame:
        """calculate pooling weight of a given date range"""
        if dates is None:
            dates = CALENDAR.slice(CALENDAR.td_within(start , end) , self.init_date , CALENDAR.updated())
        factor_weight = MarketEventMomentumFactorWeight(self.sub_factors)
        factor_weight.eval_factor_weights(min(dates) , max(dates))
        df = factor_weight.factor_weight_full.loc[dates].reset_index(['date'] ,drop = False).rename_axis(None , axis = 1)
        return df
