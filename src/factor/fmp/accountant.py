import pandas as pd
import numpy as np

from pathlib import Path
from typing import Literal , Optional , Sequence

from src.basic import INSTANCE_RECORD , CONF
from src.basic.conf import ROUNDING_RETURN , TRADE_COST
from src.data import DATAVENDOR
from src.factor.util import Portfolio , Benchmark , RISK_MODEL , Port

__all__ = ['PortfolioAccountant' , 'PortfolioAccountManager']

class PortAccount:
    def __init__(self , 
                 portfolio : Portfolio , benchmark : Benchmark | Portfolio | str , 
                 trade_engine : Literal['default' , 'harvest' , 'yale'] = 'default'):
        self.portfolio = portfolio
        self.benchmark = Benchmark(benchmark) if isinstance(benchmark , str) else benchmark

        self.port_dates = self.portfolio.available_dates()

        if trade_engine == 'default':
            self.price_type = 'close'
            self.trade_cost = 0.00035
        elif trade_engine == 'harvest':
            self.price_type = 'vwap'
            self.trade_cost = TRADE_COST
        elif trade_engine == 'yale':
            self.price_type = 'open'
            self.trade_cost = 0.00035

    def __repr__(self):
        return f'{self.__class__.__name__}(portfolio={self.portfolio},benchmark={self.benchmark})'
    
    def setup(self , start : int = -1 , end : int = 99991231 , daily = False):
        port_min , port_max = self.port_dates.min() , self.port_dates.max()
        start = np.max([port_min , start])
        end   = np.min([DATAVENDOR.td(port_max,5).td , end , DATAVENDOR.td(DATAVENDOR.last_quote_dt,-1).td])

        self.model_dates = DATAVENDOR.td_within(start , end)
        if len(self.model_dates) == 0:
            self.account = pd.DataFrame()
            return self
            
        if daily:
            self.period_st = DATAVENDOR.td_array(self.model_dates , 1)
            self.period_ed = self.period_st
        else:
            self.model_dates = np.intersect1d(self.model_dates , self.port_dates)
            self.period_st = DATAVENDOR.td_array(self.model_dates , 1)
            self.period_ed = np.concatenate([self.model_dates[1:] , [DATAVENDOR.td(end,1).td]])

        assert np.all(self.model_dates < self.period_st) , (self.model_dates , self.period_st)
        assert np.all(self.period_st <= self.period_ed) , (self.period_st , self.period_ed)

        self.account = pd.DataFrame({
            'model_date':np.concatenate([[-1],self.model_dates]) , 
            'start':np.concatenate([[self.model_dates[0]],self.period_st]) , 
            'end':np.concatenate([[self.model_dates[0]],self.period_ed]) ,
            'pf':0. , 'bm':0. , 'turn':0. , 'excess':0. ,
            'analytic':None , 'attribution':None}).set_index('model_date').sort_index()
        
        return self

    def make(self , analytic = True , attribution = True , index : dict = {}):
        if self.account.empty: return self
        port_old = Port.none_port(self.model_dates[0])
        for date , ed in zip(self.model_dates , self.period_ed):
            port_new = self.portfolio.get(date) if self.portfolio.has(date) else port_old
            bench = self.benchmark.get(date , True)

            pf_ret = self.port_ret(port_old , port_new , ed , self.price_type)
            bm_ret = self.bench_ret(bench , ed)
            turn = port_new.turnover(port_old)

            self.account.loc[date , ['pf' , 'bm' , 'turn']] = [pf_ret , bm_ret , turn]
            
            if analytic: 
                self.account.loc[date , 'analytic']    = RISK_MODEL.get(date).analyze(port_new , bench , port_old) #type:ignore
            if attribution: 
                self.account.loc[date , 'attribution'] = RISK_MODEL.get(date).attribute(port_new , bench , ed , turn * self.trade_cost)  #type:ignore
            port_old = port_new.evolve_to_date(ed)

        self.account['pf']  = self.account['pf'] - self.account['turn'] * self.trade_cost
        self.account['excess'] = self.account['pf'] - self.account['bm']
        self.account = self.account.reset_index()

        self.account = self.account.assign(**index).set_index(list(index.keys())).sort_values('model_date')
        return self


    @classmethod
    def port_ret(cls , port_old : Port , port_new : Port , end : int , price_type : str = 'close'):
        if port_new is port_old or price_type == 'close':
            ret = port_new.fut_ret(end)
        elif price_type == 'open':
            ret = (port_new.fut_ret(end) + 1) * (port_old.close2open() + 1) / (port_new.close2open() + 1) - 1  
        elif price_type == 'vwap':
            ret = (port_new.fut_ret(end) + 1) * (port_old.close2vwap() + 1) / (port_new.close2vwap() + 1) - 1
        else:
            raise ValueError(f'Unknown price type: {price_type}')
        return np.round(ret , ROUNDING_RETURN)

    @classmethod
    def bench_ret(cls , bench : Port , end : int):
        return bench.fut_ret(end)
    
    @classmethod
    def create(cls , portfolio : Portfolio , benchmark : Benchmark | Portfolio | str , 
               start : int = -1 , end : int = 99991231 , 
               analytic = True , attribution = True , index : dict = {} ,
               trade_engine : Literal['default' , 'harvest' , 'yale'] = 'default' ,
               daily = False):
        acc = cls(portfolio , benchmark , trade_engine)
        acc.setup(start , end , daily).make(analytic , attribution , index)
        return acc
    

class PortfolioAccountant:
    '''
    portfolio : Portfolio
    benchmark : Benchmark | str , must given
    daily : bool , or in portfolio dates
    analytic : bool
    attribution : bool
    '''
    def __init__(self , portfolio : Portfolio , benchmark : Optional[Portfolio | Benchmark | str] , 
                 account_path : Optional[str | Path] = None):
        self.portfolio = portfolio
        self.benchmark = self.get_benchmark(benchmark)
        self.account = pd.DataFrame()
        self.account_path = account_path

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(portfolio={self.portfolio},benchmark={self.benchmark})'

    @staticmethod
    def get_benchmark(benchmark : Optional[Portfolio | Benchmark | str] = None): 
        if benchmark is None:
            benchmark = Portfolio()
        elif isinstance(benchmark , str):
            benchmark = Benchmark(benchmark)
        return benchmark

    def accounting(self , start : int = -1 , end : int = 99991231 , 
                   analytic = True , attribution = True , index : dict = {} ,
                   trade_engine : Literal['default' , 'harvest' , 'yale'] = 'default', 
                   daily = False):
        '''Accounting portfolio through date, if resume is True, will resume from last account date'''
        account = PortAccount.create(self.portfolio , self.benchmark , 
                                     start , end , analytic , attribution , index , trade_engine , daily)
        if not account.account.empty:
            self.account = account.account
        return self
    
    @staticmethod
    def total_account(accounts : Sequence[pd.DataFrame] | dict[str,pd.DataFrame]) -> pd.DataFrame:
        if not accounts: return pd.DataFrame()
        
        if isinstance(accounts , dict):
            accounts = list(accounts.values())

        df = pd.concat(accounts)
        old_index = list(df.index.names)
        df = df.reset_index().sort_values('model_date')
        df['benchmark'] = pd.Categorical(df['benchmark'] , categories = CONF.CATEGORIES_BENCHMARKS , ordered=True) 

        df = df.set_index(old_index).sort_index()
        INSTANCE_RECORD.update_account(df)    
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
            if df.empty: continue
            if name in self.accounts:
                old_df = self.accounts[name][self.accounts[name]['model_date'] < df['model_date'].min()]
                self.accounts[name] = pd.concat([old_df , df]).sort_values('model_date')
            else:
                self.accounts[name] = df
        return self
    
    @property
    def account_names(self):
        return list(self.accounts.keys())
    
    def account_last_model_dates(self):
        return {name:df['model_date'].max() for name,df in self.accounts.items() if not df.empty}
    
    def account_last_end_dates(self):
        return {name:df['end'].max() for name,df in self.accounts.items() if not df.empty}

    def load_single(self , path : str | Path , missing_ok = True , append = True):
        path = Path(path)
        assert missing_ok or path.exists() , f'{path} not exist'
        df = pd.read_pickle(path) 
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
        if fmp_names is None: fmp_names = list(self.accounts.keys())
        fmp_paths = {name:self.account_dir.joinpath(f'{name}.pkl') for name in fmp_names}
        if not overwrite:
            existed = [path for path in fmp_paths.values() if path.exists()]
            assert not existed , f'Existed paths : {existed}'
        for name in fmp_names:
            self.accounts[name].to_pickle(fmp_paths[name])
        return self
    
    def select_analytic(self , category : Literal['optim' , 'top'] , task_name : str , **kwargs):
        from src.factor.analytic import FmpOptimManager , FmpTopManager , BaseOptimCalc , BaseTopPortCalc

        task_list = FmpTopManager.TASK_LIST if category == 'top' else FmpOptimManager.TASK_LIST
        match_task = [task for task in task_list if task.match_name(task_name)]
        assert match_task and len(match_task) <= 1 , f'no match or duplicate match tasks : {task_name}'
        task , task_name = match_task[0] , match_task[0].__name__
        if not hasattr(self , 'analytic_tasks'): self.analytic_tasks : dict[str , BaseOptimCalc | BaseTopPortCalc] = {}
        if task_name not in self.analytic_tasks: self.analytic_tasks[task_name] = task(**kwargs)
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
        
        if account.empty: return self
        task = self.select_analytic(category , task_name , **kwargs)
        task.calc(account)
        if plot: task.plot(show = display)  
        return self      