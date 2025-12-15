import pandas as pd
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Literal , Sequence , Any

from src.proj import InstanceRecord
from src.basic import CONF
from src.data import DATAVENDOR
from src.res.factor.util import Portfolio , Benchmark , RISK_MODEL , Port

__all__ = ['PortfolioAccountant' , 'PortfolioAccountManager']

@dataclass(frozen=True)
class AccountConfig:              
    benchmark : Portfolio
    start : int = -1
    end : int = 99991231
    analytic : bool = True
    attribution : bool = True
    trade_engine : str = 'default'
    daily : bool = False
    
    @staticmethod
    def get_benchmark(benchmark : Portfolio | Benchmark | str | None = None) -> Portfolio: 
        if benchmark is None:
            benchmark = Portfolio()
        elif isinstance(benchmark , str):
            benchmark = Benchmark(benchmark)
        else:
            assert isinstance(benchmark , Portfolio) , f'benchmark must be Portfolio or Benchmark or str'
        return benchmark

    @property
    def price_type(self): 
        if self.trade_engine == 'default':
            return 'close'
        elif self.trade_engine == 'harvest':
            return 'vwap'
        elif self.trade_engine == 'yale':
            return 'open'
        else:
            raise ValueError(f'Unknown trade engine: {self.trade_engine}')
    @property
    def trade_cost(self): 
        if self.trade_engine == 'default':
            return CONF.Factor.TRADE.default
        elif self.trade_engine == 'harvest':
            return CONF.Factor.TRADE.harvest
        elif self.trade_engine == 'yale':
            return CONF.Factor.TRADE.yale
        else:
            raise ValueError(f'Unknown trade engine: {self.trade_engine}')

class PortfolioAccountant:
    '''
    portfolio : Portfolio
    benchmark : Benchmark | str , must given
    daily : bool , or in portfolio dates
    analytic : bool
    attribution : bool
    '''
    _instances = {}

    def __new__(cls, portfolio: Portfolio):
        key = id(portfolio)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]
    
    def __init__(self , portfolio : Portfolio):
        self.portfolio = portfolio
        self.stored_accounts : dict[AccountConfig , pd.DataFrame] = {}
        self.account = pd.DataFrame()
        self.config : AccountConfig | None = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(portfolio={self.portfolio})'
    
    def clear(self):
        self.stored_accounts.clear()
        self.account = pd.DataFrame()
        self.config = None
        return self
    @property
    def port_dates(self):
        return self.portfolio.available_dates()
    
    def setup(self):
        assert self.config is not None , 'config is not set'
        if len(self.port_dates) == 0:
            self.account = pd.DataFrame()
            return self
        
        port_min , port_max = self.port_dates.min() , self.port_dates.max()
        start = np.max([port_min , self.config.start])
        end   = np.min([DATAVENDOR.td(port_max,5).td , self.config.end , DATAVENDOR.td(DATAVENDOR.last_quote_dt,-1).td])

        model_dates = DATAVENDOR.td_within(start , end)
        if len(model_dates) == 0:
            self.account = pd.DataFrame()
            return self
            
        if self.config.daily:
            period_st = DATAVENDOR.td_array(model_dates , 1)
            period_ed = period_st
        else:
            model_dates = np.intersect1d(model_dates , self.port_dates)
            period_st = DATAVENDOR.td_array(model_dates , 1)
            period_ed = np.concatenate([model_dates[1:] , [DATAVENDOR.td(end,1).td]])

        assert np.all(model_dates < period_st) , (model_dates , period_st)
        assert np.all(period_st <= period_ed) , (period_st , period_ed)

        self.account = pd.DataFrame({
            'model_date':np.concatenate([[-1],model_dates]) , 
            'start':np.concatenate([[model_dates[0]],period_st]) , 
            'end':np.concatenate([[model_dates[0]],period_ed]) ,
            'pf':0. , 'bm':0. , 'turn':0. , 'excess':0. , 'overnight': 0. ,
            'analytic':None , 'attribution':None}).set_index('model_date').sort_index()
        
        return self
    
    def make(self):
        assert self.config is not None , 'config is not set'
        if self.account.empty: 
            return self
            
        port_old = Port.none_port(self.account.index.values[1])
        for date , ed in zip(self.account.index.values[1:] , self.account['end'].values[1:]):
            port_new = self.portfolio.get(date) if self.portfolio.has(date) else port_old
            bench = self.config.benchmark.get(date , True)

            rets = self.get_rets(port_old , port_new , bench , ed)
            turn = port_new.turnover(port_old)
            self.account.loc[date , ['pf' , 'bm' , 'overnight' , 'turn']] = \
                np.round([rets['pf'] , rets['bm'] , rets['overnight'] , turn] , CONF.Factor.ROUNDING.ret)
            
            if self.config.analytic: 
                self.account.loc[date , 'analytic']    = RISK_MODEL.get(date).analyze(port_new , bench , port_old) #type:ignore
            if self.config.attribution: 
                self.account.loc[date , 'attribution'] = RISK_MODEL.get(date).attribute(port_new , bench , ed , turn * self.config.trade_cost)  #type:ignore
            port_old = port_new.evolve_to_date(ed)

        self.account['pf']  = self.account['pf'] - self.account['turn'] * self.config.trade_cost
        self.account['excess'] = self.account['pf'] - self.account['bm']
        self.account = self.account.reset_index()

        return self

    def get_rets(self , port_old : Port , port_new : Port , bench : Port , end : int) -> dict[str,float]:
        assert self.config is not None , 'config is not set'
        
        rets : dict[str,float] = {}
        fut_ret = port_new.fut_ret(end)
        bm_ret = bench.fut_ret(end)
        overnight_ret = 0.
        if port_new is port_old or self.config.price_type == 'close':
            ...
        elif self.config.price_type == 'open':
            overnight_ret = (port_old.close2open() + 1) - 1
            shadow_overnight = (port_new.close2open() + 1) - 1
            fut_ret = (fut_ret + 1) * (overnight_ret + 1) / (shadow_overnight + 1) - 1      
        elif self.config.price_type == 'vwap':
            overnight_ret = (port_old.close2vwap() + 1) - 1
            shadow_overnight = (port_new.close2vwap() + 1) - 1
            fut_ret = (fut_ret + 1) * (overnight_ret + 1) / (shadow_overnight + 1) - 1
        else:
            raise ValueError(f'Unknown price type: {self.config.price_type}')
        rets['pf'] = fut_ret
        rets['bm'] = bm_ret
        rets['overnight'] = overnight_ret
        return rets

    def accounting(self , 
                   config_or_benchmark : AccountConfig | Portfolio | Benchmark | str | None = None ,
                   start : int = -1 , end : int = 99991231 , 
                   analytic = True , attribution = True , 
                   trade_engine : Literal['default' , 'harvest' , 'yale'] | str = 'default' , 
                   daily = False , store = False):
        '''Accounting portfolio through date, if resume is True, will resume from last account date'''
        if isinstance(config_or_benchmark , AccountConfig):
            self.config = config_or_benchmark
        else:
            benchmark = AccountConfig.get_benchmark(config_or_benchmark)
            self.config = AccountConfig(benchmark , start , end , analytic , attribution , trade_engine , daily)
        
        for config in self.stored_accounts:
            if config == self.config:
                self.account = self.stored_accounts[config]
                break
        else:
            self.setup().make()
            if store: 
                self.stored_accounts[self.config] = self.account
        return self
    
    def account_with_index(self , add_index : dict[str,Any] | None = None):
        add_index = add_index or {}
        if not add_index: 
            return self.account
        return self.account.assign(**add_index).set_index(list(add_index.keys())).sort_values('model_date')
    
    @staticmethod
    def total_account(accounts : Sequence[pd.DataFrame] | dict[str,pd.DataFrame]) -> pd.DataFrame:
        if not accounts: 
            return pd.DataFrame()
        
        if isinstance(accounts , dict):
            accounts = list(accounts.values())

        df = pd.concat(accounts)
        old_index = list(df.index.names)
        df = df.reset_index().sort_values('model_date')
        new_bm = np.setdiff1d(df['benchmark'] , CONF.Factor.BENCH.categories).tolist()
        df['benchmark'] = pd.Categorical(df['benchmark'] , categories = CONF.Factor.BENCH.categories + new_bm , ordered=True) 

        df = df.set_index(old_index).sort_index()
        InstanceRecord.update_account(df)    
        return df
    
class PortfolioAccountManager:   
    def __init__(self , account_dir : str | Path):
        self.account_dir = Path(account_dir)
        self.accounts : dict[str , pd.DataFrame] = {}
        self.account_dir.mkdir(exist_ok=True)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.account_names})'
    
    def append_accounts(self , **accounts : pd.DataFrame):
        for name , df in accounts.items():
            if df.empty: 
                continue
            if name in self.accounts:
                self.accounts[name] = pd.concat([self.accounts[name] , df]).drop_duplicates(subset=['model_date'] , keep='last').sort_values('model_date')
            else:
                self.accounts[name] = df
        return self
    
    @property
    def account_names(self):
        return list(self.accounts.keys())
    
    def account_last_model_dates(self):
        return {name:np.max(df['model_date'].to_numpy(int)) for name,df in self.accounts.items() if not df.empty}
    
    def account_last_end_dates(self):
        return {name:np.max(df['end'].to_numpy(int)) for name,df in self.accounts.items() if not df.empty}

    def load_single(self , path : str | Path , missing_ok = True , append = True):
        path = Path(path)
        assert missing_ok or path.exists() , f'{path} not exist'
        df : pd.DataFrame | Any = pd.read_pickle(path) 
        if path.stem in self.accounts:
            if append:
                df = pd.concat([df , self.accounts[path.stem]]).drop_duplicates(subset=['model_date'])
            else:
                raise KeyError(f'{path.stem} is already in the accounts')
        self.accounts[path.stem] = df
        return self
    
    def clear(self):
        self.accounts.clear()
        return self
    
    def load_dir(self , append = True):
        [self.load_single(path , append = append) for path in self.account_dir.iterdir() if path.suffix == '.pkl']
        return self
    
    def deploy(self , fmp_names : list[str] | None = None , overwrite = False):
        if fmp_names is None: 
            fmp_names = list(self.accounts.keys())
        fmp_paths = {name:self.account_dir.joinpath(f'{name}.pkl') for name in fmp_names}
        if not overwrite:
            existed = [path for path in fmp_paths.values() if path.exists()]
            assert not existed , f'Existed paths : {existed}'
        for name in fmp_names:
            self.accounts[name].to_pickle(fmp_paths[name])
        return self
    
    def select_analytic(self , category : Literal['optim' , 'top'] , task_name : str , **kwargs):
        from src.res.factor.analytic import OptimFMPTest , TopFMPTest , BaseFactorAnalyticCalculator

        task_list = TopFMPTest.TASK_LIST if category == 'top' else OptimFMPTest.TASK_LIST
        match_task = [task for task in task_list if task.match_name(task_name)]
        assert match_task and len(match_task) <= 1 , f'no match or duplicate match tasks : {task_name}'
        task , task_name = match_task[0] , match_task[0].__name__
        if not hasattr(self , 'analytic_tasks'): 
            self.analytic_tasks : dict[str , BaseFactorAnalyticCalculator] = {}
        if task_name not in self.analytic_tasks: 
            self.analytic_tasks[task_name] = task(**kwargs)
        return self.analytic_tasks[task_name]
    
    def analyze(self , category : Literal['optim' , 'top'] ,
                task_name : Literal['FrontFace', 'Perf_Curve', 'Perf_Drawdown', 'Perf_Year', 'Perf_Month',
                                    'Perf_Excess','Perf_Lag','Exp_Style','Exp_Style','Exp_Indus',
                                    'Attrib_Source','Attrib_Style'] | str , 
                plot = True , display = True , **kwargs):
        dfs = {name : df for name , df in self.accounts.items() if name.lower().startswith(category)}
        if not dfs: 
            print(f'No {category} accounts to account!')
        account = PortfolioAccountant.total_account(dfs)
        
        if account.empty: 
            return self
        task = self.select_analytic(category , task_name , **kwargs)
        task.calc(account)
        if plot: 
            task.plot(show = display)  
        return self      