import pandas as pd
import numpy as np

from pathlib import Path
from typing import Literal , Optional , Sequence

from src.basic import INSTANCE_RECORD , CONF
from src.basic.conf import ROUNDING_RETURN , ROUNDING_TURNOVER , TRADE_COST
from src.data import DATAVENDOR
from src.factor.util import Portfolio , Benchmark , RISK_MODEL , Port

from .fmp_basic import parse_full_name

def portfolio_account(portfolio : Portfolio , benchmark : Benchmark | str , 
                      start : int = -1 , end : int = 99991231 , daily = False , 
                      analytic = True , attribution = True , index : dict = {}):
    '''Accounting portfolio through date, if resume is True, will resume from last account date'''
    port_min , port_max = portfolio.available_dates().min() , portfolio.available_dates().max()
    start = np.max([port_min , start])
    end   = np.min([DATAVENDOR.td(port_max,5).td , end , DATAVENDOR.td(DATAVENDOR.last_quote_dt,-1).td])

    model_dates = DATAVENDOR.td_within(start , end)
    if len(model_dates) == 0:
        return pd.DataFrame()
        
    if daily:
        period_st = DATAVENDOR.td_array(model_dates , 1)
        period_ed = period_st
    else:
        model_dates = np.intersect1d(model_dates , portfolio.available_dates())
        period_st = DATAVENDOR.td_array(model_dates , 1)
        period_ed = np.concatenate([model_dates[1:] , [DATAVENDOR.td(end,1).td]])

    assert np.all(model_dates < period_st) , (model_dates , period_st)
    assert np.all(period_st <= period_ed) , (period_st , period_ed)

    account = pd.DataFrame({
        'model_date':np.concatenate([[-1],model_dates]) , 
        'start':np.concatenate([[model_dates[0]],period_st]) , 
        'end':np.concatenate([[model_dates[0]],period_ed]) ,
        'pf':0. , 'bm':0. , 'turn':0. , 'excess':0. ,
        'analytic':None , 'attribution':None}).set_index('model_date').sort_index()

    port_old = Port.none_port(model_dates[0])
    for date , ed in zip(model_dates , period_ed):
        port_new = portfolio.get(date) if portfolio.has(date) else port_old
        bench = benchmark.get(date , True)

        turn = np.round(port_new.turnover(port_old),ROUNDING_TURNOVER)
        account.loc[date , ['pf' , 'bm' , 'turn']] = \
            [np.round(port_new.fut_ret(ed) , ROUNDING_RETURN) , np.round(bench.fut_ret(ed) , ROUNDING_RETURN) , turn]
        
        if analytic: 
            account.loc[date , 'analytic']    = RISK_MODEL.get(date).analyze(port_new , bench , port_old) #type:ignore
        if attribution: 
            account.loc[date , 'attribution'] = RISK_MODEL.get(date).attribute(port_new , bench , ed , turn * TRADE_COST)  #type:ignore
        port_old = port_new.evolve_to_date(ed)

    account['pf']  = account['pf'] - account['turn'] * TRADE_COST
    account['excess'] = account['pf'] - account['bm']
    account = account.reset_index()

    return account.assign(**index).set_index(list(index.keys())).sort_values('model_date')

def total_account(accounts : Sequence[pd.DataFrame]):
    df = pd.concat(accounts)
    old_index = list(df.index.names)
    df = df.reset_index().sort_values('model_date')
    df['benchmark'] = pd.Categorical(df['benchmark'] , categories = CONF.CATEGORIES_BENCHMARKS , ordered=True) 

    df = df.set_index(old_index).sort_index()
    INSTANCE_RECORD.update_account(df)    
    return df


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

    def accounting(self , start : int = -1 , end : int = 99991231 , daily = False , 
                   analytic = True , attribution = True , index : dict = {}):
        '''Accounting portfolio through date, if resume is True, will resume from last account date'''
        account = portfolio_account(self.portfolio , self.benchmark , 
                                    start , end , daily , analytic , attribution , index)
        if account.empty: return account
        self.account = account
        return self
    
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

    def filter_accounts(self , **kwargs):
        dfs : dict[str,pd.DataFrame] = {}
        for name , df in self.accounts.items():
            elements = parse_full_name(name)
            if any([elements[k] != v for k,v in kwargs.items() if k in elements]):
                continue
            dfs[name] = df
        return dfs

    def total_account(self , category : Literal['optim' , 'top'] , **kwargs):
        '''
        kwargs indicate other filters , such as suffixes == ['indep'] or ['conti'] 
        '''
        dfs = self.filter_accounts(category = category , **kwargs)
        if not dfs:
            print(f'No {category} accounts to account!')
            return pd.DataFrame()
        df = total_account(list(dfs.values()))
        return df

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
        account = self.total_account(category , **kwargs)
        if account.empty: return self
        task = self.select_analytic(category , task_name , **kwargs)
        task.calc(account)
        if plot: task.plot(show = display)  
        return self      