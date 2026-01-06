import shutil
import pandas as pd
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Literal , Sequence , Any

from src.proj import Proj , Logger , CALENDAR , DB
from src.data import DATAVENDOR
from src.res.factor.util import Portfolio , Benchmark , RISK_MODEL , Port

__all__ = ['PortfolioAccountant' , 'PortfolioAccountManager']

@dataclass(frozen=True)
class AccountConfig:              
    name : str
    benchmark : Portfolio
    start : int = -1
    end : int = 99991231
    analytic : bool = True
    attribution : bool = True
    trade_engine : str = 'default'
    daily : bool = False
    indent : int = 0
    vb_level : int = 1
    
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
            return Proj.Conf.Factor.TRADE.default
        elif self.trade_engine == 'harvest':
            return Proj.Conf.Factor.TRADE.harvest
        elif self.trade_engine == 'yale':
            return Proj.Conf.Factor.TRADE.yale
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
        self.cached_accounts : dict[str , pd.DataFrame] = {}
        self.account = pd.DataFrame()
        self.config : AccountConfig | None = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(portfolio={self.portfolio})'
    
    def clear(self):
        self.cached_accounts.clear()
        self.account = pd.DataFrame()
        self.config = None
        return self
    @property
    def port_dates(self):
        return self.portfolio.available_dates()

    def accounting(self , 
                   config_or_benchmark : AccountConfig | Portfolio | Benchmark | str | None = None ,
                   start : int = -1 , end : int = 99991231 , analytic = True , attribution = True , * ,
                   trade_engine : Literal['default' , 'harvest' , 'yale'] | str = 'default' , 
                   daily = False , cache = False , indent : int = 0 , vb_level : int = 1):
        '''Accounting portfolio through date, if resume is True, will resume from last account date'''
        if isinstance(config_or_benchmark , AccountConfig):
            self.config = config_or_benchmark
        else:
            benchmark = AccountConfig.get_benchmark(config_or_benchmark)
            self.config = AccountConfig(self.portfolio.name , benchmark , start , end , analytic , attribution , 
                                        trade_engine = trade_engine , daily = daily ,
                                        indent = indent , vb_level = vb_level)
        
        if str(self.config) in self.cached_accounts:
            self.account = self.cached_accounts[str(self.config)]
        else:
            self.setup().make()
            if cache: 
                self.cached_accounts[str(self.config)] = self.account
        return self
    
    def setup(self):
        assert self.config is not None , 'config is not set'
        if len(self.port_dates) == 0:
            self.account = pd.DataFrame()
            return self
        
        port_min , port_max = self.port_dates.min() , self.port_dates.max()
        start = np.max([port_min , self.config.start])
        end   = np.min([DATAVENDOR.td(port_max,5) , self.config.end , DATAVENDOR.td(DATAVENDOR.last_quote_dt,-1)])

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
            period_ed = np.concatenate([model_dates[1:] , [DATAVENDOR.td(end,1)]])

        assert np.all(model_dates < period_st) , (model_dates , period_st)
        assert np.all(period_st <= period_ed) , (period_st , period_ed)

        self.model_dates = np.concatenate([[-1],model_dates])
        self.start_dates = np.concatenate([[model_dates[0]],period_st])
        self.end_dates   = np.concatenate([[model_dates[0]],period_ed])
        self.account = pd.DataFrame({
            'model_date':self.model_dates , 'start':self.start_dates , 'end':self.end_dates ,
            'pf':0. , 'bm':0. , 'turn':0. , 'excess':0. , 'overnight': 0. ,
            'analytic':None , 'attribution':None}).set_index('model_date').sort_index()
            
        return self
    
    def make(self):
        assert self.config is not None , 'config is not set'
        if self.account.empty: 
            return self

        port_old = Port.none_port(self.model_dates[1])
        
        Logger.stdout(f'{self.config.name} has {len(self.account) - 1} account dates at {CALENDAR.dates_str([self.start_dates[1] , self.end_dates[-1]])}' , 
                      indent = self.config.indent , vb_level = self.config.vb_level)
        for i , (mdate , ed) in enumerate(zip(self.model_dates[1:] , self.end_dates[1:])):
            port_new = self.portfolio.get(mdate) if self.portfolio.has(mdate) else port_old
            bench = self.config.benchmark.get(mdate , True)

            rets = self.get_rets(port_old , port_new , bench , ed)
            turn = port_new.turnover(port_old)
            self.account.loc[mdate , ['pf' , 'bm' , 'overnight' , 'turn']] = \
                np.round([rets['pf'] , rets['bm'] , rets['overnight'] , turn] , Proj.Conf.Factor.ROUNDING.ret)
            
            if self.config.analytic: 
                self.account.loc[mdate , 'analytic']    = RISK_MODEL.get(mdate).analyze(port_new , bench , port_old) #type:ignore
            if self.config.attribution: 
                self.account.loc[mdate , 'attribution'] = RISK_MODEL.get(mdate).attribute(port_new , bench , ed , turn * self.config.trade_cost)  #type:ignore
            port_old = port_new.evolve_to_date(ed)
            if i > 0 and ((i + 1) % 100 == 0 or i == len(self.account) - 2):
                Logger.stdout(f'{self.config.name} accounting {i + 1} / {len(self.account) - 1} at {mdate}' , 
                              indent = self.config.indent + 1 , vb_level = self.config.vb_level + 2)

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
    
    @staticmethod
    def account_with_index(account : pd.DataFrame , add_index : dict[str,Any] | None = None):
        add_index = add_index or {}
        if not add_index: 
            return account
        return account.assign(**add_index).set_index(list(add_index.keys())).sort_values('model_date').sort_index()

    @staticmethod
    def concat_accounts(accounts : Sequence[pd.DataFrame]):
        return pd.concat([acc for acc in accounts]).drop_duplicates(subset = ['model_date'] , keep = 'last').sort_values('model_date')
    
    @staticmethod
    def total_account(accounts : Sequence[pd.DataFrame] | dict[str,pd.DataFrame]) -> pd.DataFrame:
        if not accounts: 
            return pd.DataFrame()
        
        if isinstance(accounts , dict):
            accounts = list(accounts.values())

        df = pd.concat(accounts)
        old_index = [index for index in df.index.names if index]
        df = df.reset_index(old_index , drop = False).sort_values('model_date').reset_index(drop = True)
        new_bm = np.setdiff1d(df['benchmark'] , Proj.Conf.Factor.BENCH.categories).tolist()
        df['benchmark'] = pd.Categorical(df['benchmark'] , categories = Proj.Conf.Factor.BENCH.categories + new_bm , ordered=True) 

        df = df.set_index(old_index).sort_index()
        Proj.States.account = df    
        return df

    @classmethod
    def account_to_dfs(cls , account : pd.DataFrame) -> dict[str,pd.DataFrame]:
        account = account.set_index('model_date')
        dfs = {}
        dfs['basic'] = account.loc[:,['start' , 'end' , 'pf' , 'bm' , 'turn' , 'excess' , 'overnight']].reset_index(drop = False)
        
        dfs.update(RISK_MODEL.Analytics_to_dfs(account['analytic'].to_dict()))
        dfs.update(RISK_MODEL.Attributions_to_dfs(account['attribution'].to_dict()))
        return dfs

    @classmethod
    def dfs_to_account(cls , dfs : dict[str,pd.DataFrame]) -> pd.DataFrame:
        if not dfs or 'basic' not in dfs:
            return pd.DataFrame()
        account = dfs['basic'].assign(analytic=None, attribution=None).set_index('model_date')
        analytics = RISK_MODEL.Analytics_from_dfs({k:v for k,v in dfs.items() if k.startswith('analytic_')})
        attributions = RISK_MODEL.Attributions_from_dfs({k:v for k,v in dfs.items() if k.startswith('attribution_')})
        for date , v in analytics.items():
            account.loc[date , 'analytic'] = v #type:ignore
        for date , v in attributions.items():
            account.loc[date , 'attribution'] = v #type:ignore
        return account.reset_index(drop = False)

    def save(self , path : Path | str | None = None):
        self.save_account(self.account , path)
        return self

    @classmethod
    def save_account(cls , account : pd.DataFrame , path : Path | str | None = None , vb_level : int = 1 , indent : int = 0):
        if path is None or account.empty:
            return
        path = Path(path)
        assert not path.exists() or path.is_dir() , f'{path} is a file'
        account_dfs = cls.account_to_dfs(account)
        if path.exists():
            shutil.rmtree(path)
            status = 'Overwritten '
        else:
            status = 'File Created'
        path.mkdir(parents=True, exist_ok=True)
        for name , df in account_dfs.items():
            DB.save_df(df , path.joinpath(f'{name}.feather') , prefix = 'Account' , vb_level = 99)
        Logger.stdout(f'Account {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        
    @classmethod
    def load(cls , path : Path | str | None = None) -> pd.DataFrame:
        if path is None:
            return pd.DataFrame()
        path = Path(path)
        assert not path.exists() or path.is_dir() , f'{path} is not a directory'
        account_dfs = {path.stem:DB.load_df(path) for path in path.glob('*.feather')}
        return cls.dfs_to_account(account_dfs)

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
            Logger.stdout(f'No {category} accounts to account!')
        account = PortfolioAccountant.total_account(dfs)
        
        if account.empty: 
            return self
        task = self.select_analytic(category , task_name , **kwargs)
        task.calc(account)
        if plot: 
            task.plot(show = display)  
        return self      