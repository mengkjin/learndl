'''
In this module, we will define the portfolio conditioner
The purpose of the portfolio conditioner is to adjust the portfolio according to conditions,
for example, 
1. the drawdown is too high, we need cut the position to 0
2. the rebounce is sufficient, we add back the position to 1
'''

import pandas as pd
import numpy as np
from abc import ABC , abstractmethod
from dataclasses import dataclass
from typing import Any

from src.proj import Proj
from src.res.factor.util import Portfolio

from ..stat import eval_cum_ret , eval_drawdown , eval_uncovered_max_drawdown

def select_conditioner(name : str):
    '''
    select the conditioner based on the name
    '''
    subclasses = BaseConditioner.__subclasses__()
    candidates = [subclass for subclass in subclasses if subclass.match(name)]
    if len(candidates) == 0:
        raise ValueError(f'Conditioner {name} not found')
    elif len(candidates) > 1:
        raise ValueError(f'Conditioner {name} is ambiguous, candidates are {candidates}')
    else:
        return candidates[0]

class AccountAccessor:
    def __init__(self , input : 'Portfolio|pd.DataFrame|pd.Series|np.ndarray|AccountAccessor'):
        self.input = input
        self.account = self.get_account()

    def get_account(self) -> pd.DataFrame:
        if isinstance(self.input , Portfolio):
            df = self.input.account
        elif isinstance(self.input , pd.DataFrame):
            df = self.input
        elif isinstance(self.input , pd.Series) or isinstance(self.input , np.ndarray):
            if isinstance(self.input , np.ndarray):
                df = pd.DataFrame(self.input , columns = pd.Index(['pf']))
            else:
                df = self.input.rename('pf').to_frame()
            df = df.assign(
                start = np.arange(len(df)) , end = np.arange(len(df)) , 
                bm = 0. , turn = 0. , excess = df['pf'] , overnight = 0. ,
                analytic = None , attribution = None
            )
        elif isinstance(self.input , AccountAccessor):
            df = self.input.account
        else:
            raise ValueError(f'Unknown input type: {type(self.input)}')
        return df
    
    @property
    def index(self) -> pd.Index:
        return self.account.index
    
    @property
    def pf(self) -> pd.Series:
        return self.account.pf
    
    @property
    def bm(self) -> pd.Series:
        return self.account.bm
    
    @property
    def turn(self) -> pd.Series:
        return self.account.turn
    
    @property
    def excess(self) -> pd.Series:
        return self.account.excess
    
    @property
    def overnight(self) -> pd.Series:
        return self.account.overnight
    
    @property
    def intraday(self) -> pd.Series:
        return (1 + self.pf) / (1 + self.overnight) - 1
    
    @property
    def analytic(self) -> pd.Series:
        return self.account.analytic
    
    @property
    def attribution(self) -> pd.Series:
        return self.account.attribution
    
    @property
    def start(self) -> pd.Series:
        return self.account.start
    
    @property
    def end(self) -> pd.Series:
        return self.account.end
    
    @property
    def drawdown(self) -> pd.Series:
        drawdown = eval_drawdown(self.pf)
        assert isinstance(drawdown , pd.Series) , 'drawdown must be pd.Series'
        return drawdown
    
@dataclass
class AccountConditioner:
    accessor : AccountAccessor
    conditions : pd.Series

    @property
    def position_end(self) -> pd.Series:
        pe : pd.Series | Any = self.conditions.shift(1).fillna(1)
        return pe
    
    @property
    def conditioned_position(self) -> pd.Series:
        return self.position_end.query('position_end != 1')
    
    @property
    def position_start(self) -> pd.Series:
        ps : pd.Series | Any = self.position_end.shift(1).fillna(1)
        return ps
    
    @property
    def position_change(self) -> pd.Series:
        return self.position_end - self.position_start
    
    @property
    def unconditioned_pf(self) -> pd.Series:
        return self.accessor.pf
    
    @property
    def conditioned_pf(self) -> pd.Series:
        if (self.accessor.overnight == 0).all():
            pf = self.position_end * self.accessor.pf
        else:
            pf = (self.position_start * self.accessor.overnight + 1) * \
                (self.position_end * self.accessor.intraday + 1) - 1
        pf = pf - self.position_change.abs() * Proj.Conf.Factor.TRADE.default
        return pf
    
class BaseConditioner(ABC):
    '''
    the base abstract class of the conditioner
    '''
    CONDITIONER_NAME : str | list[str] | None = None

    def __init__(self , portfolio : Portfolio | None = None):
        assert self.CONDITIONER_NAME is not None , 'CONDITIONER_NAME must be set'
        self.portfolio = portfolio

    @classmethod
    def match(cls , name : str):
        if isinstance(cls.CONDITIONER_NAME , str):
            return name == cls.CONDITIONER_NAME
        elif isinstance(cls.CONDITIONER_NAME , list):
            return name in cls.CONDITIONER_NAME
        else:
            return False
        
    @staticmethod
    def drawdown_rebound_condition(accessor : AccountAccessor , stop_threshold : float = 0.1 , rebounce_threshold : float = 0.25) -> pd.Series:
        '''
        the basic drawdown rebound condition
            stop_threshold : the threshold of the stop to change the condition to 0
            rebounce_threshold : the threshold of the rebounce from bottom to change the condition to 1
        '''
        ret = accessor.pf
        drawdown = eval_drawdown(ret , how = 'exp')
        umd = eval_uncovered_max_drawdown(drawdown) # uncovered max drawdown
        rebounce = drawdown - umd.fillna(1)
        condition = ret * 0 + 1
        for i , (dd , uu , rb) in enumerate(zip(drawdown.values , umd.values , rebounce.values)):
            if uu < -stop_threshold and rb < rebounce_threshold:
                condition.iloc[i] = 0       
        return condition
    
    @staticmethod
    def progressive_condition(
            accessor : AccountAccessor , 
            stop_threshold : float = 0.1 , 
            rebounce_threshold : float = 0.25 ,
            progressive_add_stop : float | None = None ,
        ) -> pd.Series:
        '''
        the progressive condition
            stop_threshold : the threshold of the stop to change the condition to 0
            rebounce_threshold : the threshold of the rebounce from bottom to change the condition to 1
            progressive_add_stop : additive stop to change the stop_threshold every rebounce , 
                    if None, will use the rebounce_threshold to change the stop_threshold
        '''
        ret = accessor.pf
        drawdown = eval_drawdown(ret , how = 'exp')
        umd = eval_uncovered_max_drawdown(drawdown) # uncovered max drawdown
        recover_ratio = (1 - drawdown / umd.fillna(1))
        condition = ret * 0 + 1
        init_stop = stop_threshold
        stopped = False
        for i , (dd , uu , rr) in enumerate(zip(drawdown.values , umd.values , recover_ratio.values)):      
            if uu < -init_stop:
                if rr < rebounce_threshold:
                    stopped = True
                    condition.iloc[i] = 0
                elif stopped:
                    stopped = False
                    if progressive_add_stop is not None:
                        init_stop += progressive_add_stop
                    else:
                        init_stop += init_stop * rebounce_threshold
            if dd >= 0:
                init_stop = stop_threshold          
        return condition
    
    @classmethod    
    def conditioner_name(cls):
        assert cls.CONDITIONER_NAME is not None , 'CONDITIONER_NAME must be set'
        return cls.CONDITIONER_NAME if isinstance(cls.CONDITIONER_NAME , str) else cls.CONDITIONER_NAME[0]

    @abstractmethod
    def conditions(self , accessor : AccountAccessor) -> pd.Series:
        '''
        input : AccountAccessor
            the account accessor of the portfolio
        return : condition signal pd.Series , latent magnified factor, on the exact date of the input
        '''
        ...
    
    def conditioning(self , input : Portfolio | pd.DataFrame | pd.Series | np.ndarray | AccountAccessor | None = None):
        '''
        input : Portfolio | pd.DataFrame | pd.Series | AccountAccessor | None
            if input is None, will use the self.portfolio's account
            if input is Portfolio, will use the input's account
            if input is pd.DataFrame, it should be the account of the portfolio
            if input is pd.Series, it should be the daily pf return of the portfolio
            if input is np.ndarray, it should be the daily pf return of the portfolio
            if input is AccountAccessor, it should be the account of the accessor
        return : pd.Series , designated magnified factor, on the exact date of the input
        '''
        account_accessor = self.account_accessor if input is None else AccountAccessor(input)
        conditions = self.conditions(account_accessor)
        return AccountConditioner(account_accessor , conditions)
    
    def conditioned_portfolio(self):
        assert self.portfolio is not None , 'portfolio is not set'
        return self.portfolio.replace(self())
    
    def conditioned_pf_ret(self , input : Portfolio | pd.DataFrame | pd.Series | np.ndarray | AccountAccessor | None = None , 
                           plot = False):
        '''
        return the conditioned comparison pf return
        '''
        conditioner = self.conditioning(input)
        cum_cond_pf = eval_cum_ret(conditioner.conditioned_pf , 'exp')
        assert isinstance(cum_cond_pf , pd.Series) , 'cum_cond_pf must be pd.Series'
        cum_cond_pf.name = self.conditioner_name()
        if plot: 
            cum_cond_pf.plot(figsize = (12 , 4))

        return cum_cond_pf
    
    @staticmethod
    def select_conditioner(name : str):
        return select_conditioner(name)
        
    @property
    def account_accessor(self):
        assert self.portfolio is not None , 'portfolio is not set'
        if hasattr(self , '_account_accessor'):
            return self._account_accessor
        else:
            self._account_accessor = AccountAccessor(self.portfolio)
            return self._account_accessor
    
    @property
    def account(self):
        return self.account_accessor.account
    
    def adjusted_portfolio(self , portfolio : Portfolio):
        conditioner = self.conditioning(portfolio)
        position = conditioner.conditioned_position
        ports = [portfolio.ports[date].magnify(position.loc[date]) for date in position.index]
        return Portfolio.from_ports(*ports)

    def __call__(self) -> Portfolio:
        assert self.portfolio is not None , 'portfolio is not set'
        return self.adjusted_portfolio(self.portfolio)

class BalanceOptimisticConditioner(BaseConditioner):
    '''
    the default conditioner, which is the balance_neutral conditioner
    '''
    CONDITIONER_NAME = ['balance', 'balance_optimistic']
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        return self.drawdown_rebound_condition(accessor , stop_threshold = 0.1 , rebounce_threshold = 0.025)
    
class BalanceNeutralConditioner(BaseConditioner):
    CONDITIONER_NAME = 'balance_neutral'
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        return self.drawdown_rebound_condition(accessor , stop_threshold = 0.1 , rebounce_threshold = 0.0375)
    
class BalancePessimisticConditioner(BaseConditioner):
    CONDITIONER_NAME = 'balance_pessimistic'
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        return self.drawdown_rebound_condition(accessor , stop_threshold = 0.1 , rebounce_threshold = 0.05)
    
class ConservativeOptimisticConditioner(BaseConditioner):
    CONDITIONER_NAME = ['conservative', 'conservative_optimistic']
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        return self.drawdown_rebound_condition(accessor , stop_threshold = 0.05 , rebounce_threshold = 0.0125)
    
class ConservativeNeutralConditioner(BaseConditioner):
    CONDITIONER_NAME = 'conservative_neutral'
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        return self.drawdown_rebound_condition(accessor , stop_threshold = 0.05 , rebounce_threshold = 0.01875)
    
class ConservativePessimisticConditioner(BaseConditioner):
    CONDITIONER_NAME = 'conservative_pessimistic'
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        return self.drawdown_rebound_condition(accessor , stop_threshold = 0.05 , rebounce_threshold = 0.025)
    
class RadicalOptimisticConditioner(BaseConditioner):
    CONDITIONER_NAME = ['radical', 'radical_optimistic']
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        return self.drawdown_rebound_condition(accessor , stop_threshold = 0.2 , rebounce_threshold = 0.05)
    
class RadicalNeutralConditioner(BaseConditioner):
    CONDITIONER_NAME = 'radical_neutral'
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        return self.drawdown_rebound_condition(accessor , stop_threshold = 0.2 , rebounce_threshold = 0.075)
    
class RadicalPessimisticConditioner(BaseConditioner):
    CONDITIONER_NAME = 'radical_pessimistic'
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        return self.drawdown_rebound_condition(accessor , stop_threshold = 0.2 , rebounce_threshold = 0.1)

    
class ProgressiveConditioner(BaseConditioner):
    def conditions(self , accessor : AccountAccessor) -> pd.Series | Any:
        ret = accessor.pf
        drawdown = eval_drawdown(ret , how = 'exp')
        umd = eval_uncovered_max_drawdown(drawdown) # uncovered max drawdown
        recover_ratio = (1 - drawdown / umd.fillna(1))
        condition = ret * 0 + 1
        init_stop = 0.1
        add_stop = 0.05
        stop_threshold = init_stop
        rebounce_threshold = 0.25
        stopped = False
        for i , (dd , uu , rr) in enumerate(zip(drawdown.values , umd.values , recover_ratio.values)):
            if uu < -stop_threshold:
                if rr < rebounce_threshold:
                    stopped = True
                    condition.iloc[i] = 0
                elif stopped:
                    stopped = False
                    stop_threshold = stop_threshold + add_stop
            if dd >= 0:
                stop_threshold = init_stop          
        return condition
