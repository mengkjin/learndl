from typing import Literal
from pathlib import Path
import pandas as pd

from src.basic import  INSTANCE_RECORD , CONF

from src.factor.fmp import parse_full_name
from src.factor.analytic import FmpOptimManager , FmpTopManager
from src.factor.analytic.fmp_optim.calculator import BaseOptimCalc
from src.factor.analytic.fmp_top.calculator import BaseTopPortCalc

class PortfolioAccountManager:   
    OPT_TASK_LIST = FmpOptimManager.TASK_LIST
    TOP_TASK_LIST = FmpTopManager.TASK_LIST

    def __init__(self , account_dir : str | Path):
        self.account_dir = Path(account_dir)
        self.accounts : dict[str , pd.DataFrame] = {}
        self.tasks : dict[str , BaseOptimCalc | BaseTopPortCalc] = {}
    
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
    
    @property
    def last_account_dates(self):
        return {name:df['model_date'].max() for name,df in self.accounts.items() if not df.empty}

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
        df = pd.concat(dfs.values())
        old_index = list(df.index.names)
        df = df.reset_index().sort_values('model_date')
        df['benchmark'] = pd.Categorical(df['benchmark'] , categories = CONF.CATEGORIES_BENCHMARKS , ordered=True) 

        df = df.set_index(old_index).sort_index()
        INSTANCE_RECORD['account'] = df
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
    
    def deploy(self , overwrite = False):
        fmp_paths = {p:self.account_dir.joinpath(p) for p in self.accounts}
        if not overwrite:
            existed = [path for path in fmp_paths.values() if path.exists()]
            assert not existed , f'Existed paths : {existed}'
        for p in self.accounts:
            self.accounts[p].to_pickle(fmp_paths[p])
        return self
    
    def select_analytic(self , category : Literal['optim' , 'top'] , task_name : str):
        task_list = self.TOP_TASK_LIST if category == 'top' else self.OPT_TASK_LIST
        match_tasks = [task for task in task_list if task.match_name(task_name)]
        assert match_tasks , f'no match tasks : {task_name}'
        assert len(match_tasks) <= 1, f'Duplicate match tasks: {match_tasks}'
        use_name = match_tasks[0].__name__
        if use_name not in self.tasks:
             self.tasks[use_name] = match_tasks[0]()
        return self.tasks[use_name]

    def analyze(self , category : Literal['optim' , 'top'] ,
                task_name : Literal['FrontFace', 'Perf_Curve', 'Perf_Drawdown', 'Perf_Year', 'Perf_Month',
                                    'Perf_Excess','Perf_Lag','Exp_Style','Exp_Style','Exp_Indus',
                                    'Attrib_Source','Attrib_Style'] | str , 
                plot = True , display = True , **kwargs):
        account = self.total_account(category , **kwargs)
        if account.empty: return self
        task = self.select_analytic(category , task_name)
        task.calc(account)
        if plot: task.plot(show = display)  
        return self      