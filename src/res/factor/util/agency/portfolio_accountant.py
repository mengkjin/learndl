import pandas as pd
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Literal , Any

from src.proj import Proj , Logger , CALENDAR , DB
from src.data import DATAVENDOR
from src.res.factor.util import Portfolio , Benchmark , RISK_MODEL , Port
from src.res.factor.util.stat.aggregate import eval_period_ret

from ..stat import eval_drawdown

__all__ = ['PortfolioAccountant' , 'PortfolioAccountManager']

@dataclass(frozen=True)
class AccountConfig:              
    name : str = ''
    benchmark : str = 'default'
    start : int = -1
    end : int = 99991231
    analytic : bool = True
    attribution : bool = True
    trade_engine : str = 'default'
    daily : bool = False

    def __bool__(self):
        return True

    @property
    def key(self):
        return str(self)

    @property
    def bool(self):
        return bool(self.name)
    
    @staticmethod
    def get_benchmark(benchmark : 'AccountConfig | Portfolio | Benchmark | str | None' = None) -> Portfolio: 
        if benchmark is None:
            benchmark = Portfolio()
        elif isinstance(benchmark , AccountConfig):
            benchmark = Benchmark(benchmark.benchmark)
        elif isinstance(benchmark , str):
            benchmark = Benchmark(benchmark)
        else:
            assert isinstance(benchmark , Portfolio) , f'benchmark must be Portfolio or Benchmark or str'
        return benchmark

    @staticmethod
    def get_benchmark_name(benchmark : 'AccountConfig | Portfolio | Benchmark | str | None' = None) -> str:
        if benchmark is None:
            return 'default'
        elif isinstance(benchmark , AccountConfig):
            return benchmark.benchmark
        elif isinstance(benchmark , (Portfolio , Benchmark)):
            return benchmark.name
        elif isinstance(benchmark , str):
            return benchmark
        else:
            raise ValueError(f'Unknown benchmark type: {type(benchmark)}')

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

class PortfolioAccount:
    columns_basic = ['model_date' , 'start' , 'end' , 'pf' , 'bm' , 'turn' , 'excess' , 'overnight']
    columns_all = columns_basic + ['analytic' , 'attribution']

    def __new__(cls , input = None , *args , **kwargs):
        instance = super().__new__(cls)
        Proj.States.account = instance 
        return instance

    def __init__(self , input : 'Portfolio|pd.DataFrame|pd.Series|np.ndarray|list[float]|None' = None , 
                 config : AccountConfig | None = None , index : dict[Any,Any] | None = None):
        if isinstance(input , Portfolio):
            config = input.account.config
            index = input.account.index
            input = input.account.input
        self.input = input
        self.config = config or AccountConfig()
        self.index = index or {}

    def __repr__(self):
        return repr(self.df)

    @classmethod
    def empty_df(cls) -> pd.DataFrame:
        return pd.DataFrame(columns = cls.columns_all)

    @property
    def input(self) -> pd.DataFrame:
        return self._input
    
    @input.setter
    def input(self , value : 'pd.DataFrame|pd.Series|np.ndarray|list[float]|None' = None):
        if value is None or isinstance(value , pd.DataFrame):
            df = value
        else:
            if isinstance(value , pd.Series):
                pf = value.to_list()
            elif isinstance(value , np.ndarray):
                pf = value.tolist()
            else:
                pf = value
            dates = np.arange(len(pf))
            df = pd.DataFrame({
                'model_date' :  dates - 1 ,
                'start' : dates ,
                'end' : dates ,
                'pf' : pf ,
                'bm' : 0. ,
                'turn' : 0. ,
                'excess' : pf ,
                'overnight' : 0. ,
                'analytic' : None ,
                'attribution' : None
            })
        self._input = self.AddFirstRow(df)

    def clear(self):
        self.input = None
        self.config = AccountConfig()
        self.index = {}
        return self

    def with_index(self , index : dict[str,Any] | None = None):
        """return portfolio account with given index"""
        if index is not None:
            self.index = index
        return self

    def filter_dates(self , start : int | None = None , end : int | None = None):
        df = self.input
        if start is not None:
            df = df.query('model_date >= @start')
        if end is not None:
            df = df.query('model_date <= @end')
        self.input = df
        return self

    @classmethod
    def AddFirstRow(cls , df : pd.DataFrame | None = None):
        if df is None:
            df = pd.DataFrame()
        else:
            df = df.query('model_date >= 0') if not df.empty else pd.DataFrame()
        first_model_date = 0 if df.empty else df['model_date'].min()
        first_row = pd.DataFrame({
            'model_date' : [-1] ,
            'start' : first_model_date ,
            'end' : first_model_date ,
            'pf' : 0. ,
            'bm' : 0. ,
            'turn' : 0. ,
            'excess' : 0. ,
            'overnight' : 0. ,
            'analytic' : None ,
            'attribution' : None
        })
        df = pd.concat([first_row , df]).sort_values('model_date').reset_index(drop = True)
        return df

    @classmethod
    def Concat(cls , *accounts : 'PortfolioAccount | None'):
        accs = [acc for acc in accounts if acc is not None]
        if not accs:
            return cls()
        else:
            configs = [acc.config for acc in accs if acc.config]
            config = None if not configs else configs[-1]
            add_indexes = [acc.index for acc in accs if acc.index]
            index = None if not add_indexes else add_indexes[-1]
            df = pd.concat([acc.input for acc in accs]).query('model_date >= 0').drop_duplicates(subset = ['model_date'] , keep = 'last')
            return cls(df , config = config , index = index)
    
    @staticmethod
    def Total(*accounts : 'PortfolioAccount | None') -> pd.DataFrame:
        accs = [acc for acc in accounts if acc is not None]
        if not accs: 
            return pd.DataFrame()
        df = pd.concat([acc.df for acc in accs])
        old_index = [index for index in df.index.names if index]
        df = df.reset_index(old_index , drop = False).sort_values('model_date').reset_index(drop = True)
        new_bm = np.setdiff1d(df['benchmark'] , Proj.Conf.Factor.BENCH.categories).tolist()
        df['benchmark'] = pd.Categorical(df['benchmark'] , categories = Proj.Conf.Factor.BENCH.categories + new_bm , ordered=True) 

        df = df.set_index(old_index).sort_index()  
        return df

    @property
    def df(self) -> pd.DataFrame:
        df = self.input
        if self.index:
            df = df.assign(**self.index).set_index(list(self.index.keys()))
        return df

    def to_dfs(self) -> dict[str,pd.DataFrame]:
        dfs = {}
        dfs['basic'] = self.input.loc[:,self.columns_basic]
        dfs['index'] = pd.DataFrame(self.index , index=[0]).reset_index(drop=True)
        
        dfs.update(RISK_MODEL.Analytics_to_dfs(dict(zip(self.model_date,self.analytic))))
        dfs.update(RISK_MODEL.Attributions_to_dfs(dict(zip(self.model_date,self.attribution))))
        return dfs

    @classmethod
    def from_df(cls , df : pd.DataFrame):
        index = df.loc[:,df.columns.difference(cls.columns_all)].drop_duplicates()
        assert index.shape[0] == 1 , f'expect one row for index, got {index}'
        index_dict = index.iloc[0].to_dict()
        return cls(df , index = index_dict)

    @classmethod
    def from_dfs(cls , dfs : dict[str,pd.DataFrame]):
        if not dfs or 'basic' not in dfs:
            return cls()
        account = dfs['basic'].assign(analytic=None, attribution=None).set_index('model_date')
        analytics = RISK_MODEL.Analytics_from_dfs({k:v for k,v in dfs.items() if k.startswith('analytic_')})
        attributions = RISK_MODEL.Attributions_from_dfs({k:v for k,v in dfs.items() if k.startswith('attribution_')})
        index = dfs['index'].iloc[0].to_dict() if 'index' in dfs else {}
        for date , v in analytics.items():
            account.loc[date , 'analytic'] = v #type:ignore
        for date , v in attributions.items():
            account.loc[date , 'attribution'] = v #type:ignore
        return cls(account.reset_index(drop = False) , index = index)

    def save(self , path : Path | str | None = None , vb_level : int = 1 , indent : int = 0):
        if path is None or self.empty:
            return self
        path = Path(path)
        assert path.suffix in ['.pkl' , '.tar'] , f'{path} is not a pkl or tar file'
        if path.suffix == '.pkl':
            self.df.to_pickle(path)
            Logger.stdout(f'Account Saved to {path}' , indent = indent , vb_level = vb_level , italic = True)
        else:
            account_dfs = self.to_dfs()
            DB.save_dfs_to_tar(account_dfs , path , prefix = 'Account' , indent = indent , vb_level = vb_level)
        return self

    @classmethod
    def load(cls , path : Path | str | None = None) -> 'PortfolioAccount':
        """load portfolio account from a path (a tar file or a pickle file)"""
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            return cls()
        assert path.suffix in ['.pkl' , '.tar'] , f'{path} is not a pkl or tar file'
        if path.suffix == '.pkl':
            account = cls(pd.read_pickle(path))
        else:
            account = cls.from_dfs(DB.load_dfs_from_tar(path))
        return account

    @property
    def empty(self):
        return self.input.query('model_date >= 0').empty

    @property
    def model_date(self) -> pd.Series:
        return self.input.model_date
    
    @property
    def start(self) -> pd.Series:
        return self.input.start
    
    @property
    def end(self) -> pd.Series:
        return self.input.end
    
    @property
    def pf(self) -> pd.Series:
        return self.input.pf

    @property
    def bm(self) -> pd.Series:
        return self.input.bm
    
    @property
    def turn(self) -> pd.Series:
        return self.input.turn
    
    @property
    def excess(self) -> pd.Series:
        return self.input.excess
    
    @property
    def overnight(self) -> pd.Series:
        return self.input.overnight
    
    @property
    def intraday(self) -> pd.Series:
        return (1 + self.pf) / (1 + self.overnight) - 1
    
    @property
    def analytic(self) -> pd.Series:
        return self.input.analytic
    
    @property
    def attribution(self) -> pd.Series:
        return self.input.attribution
    
    @property
    def drawdown(self) -> pd.Series:
        drawdown = eval_drawdown(self.pf)
        assert isinstance(drawdown , pd.Series) , 'drawdown must be pd.Series'
        return drawdown

    @property
    def max_model_date(self) -> int:
        return int(self.model_date.max())
    
    @property
    def max_end_date(self) -> int:
        return int(self.end.max())
    
    def eval_period_ret(self) -> pd.Series:
        return eval_period_ret(self.df.reset_index().loc[:,['end' , 'pf']].rename(columns = {'end' : 'date'}))

    @classmethod
    def EvalPeriodRet(cls , paths : dict[str , str] | dict[str , Path]) -> pd.DataFrame:
        accounts = {name : cls.load(path) for name , path in paths.items() if Path(path).exists()}
        if not accounts:
            return pd.DataFrame()
        return pd.concat({name : account.eval_period_ret() for name , account in accounts.items()}, axis = 1)

class PortfolioAccountant:
    """
    portfolio accountant for a portfolio , one portfolio has only one accountant
    portfolio : Portfolio
    benchmark : Benchmark | str , must given
    daily : bool , or in portfolio dates
    analytic : bool
    attribution : bool
    """
    _instances = {}

    def __new__(cls, portfolio: Portfolio):
        key = id(portfolio)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]
    
    def __init__(self , portfolio : Portfolio):
        self.portfolio = portfolio
        self.account = PortfolioAccount()
        if not hasattr(portfolio , 'cached_accounts'):
            self.cached_accounts : dict[str , PortfolioAccount] = {}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(portfolio={self.portfolio})'
    
    def clear(self):
        self.cached_accounts.clear()
        self.account.clear()
        return self

    @property
    def port_dates(self):
        return self.portfolio.available_dates()

    @property
    def resume_path(self) -> Path | str | None:
        if not hasattr(self , '_resume_path'):
            self._resume_path = None
        return Path(self._resume_path) if self._resume_path else None

    @resume_path.setter
    def resume_path(self , value : Path | str | None):
        self._resume_path = Path(value) if value else None

    def accounting(self , 
                   config_or_benchmark : AccountConfig | Portfolio | Benchmark | str | None = None ,
                   start : int = -1 , end : int = 99991231 , analytic = True , attribution = True , * ,
                   trade_engine : Literal['default' , 'harvest' , 'yale'] | str = 'default' , 
                   daily = False , cache = False , with_index : dict[str,Any] | None = None ,
                   resume_path : Path | str | None = None , resume_end : int | None = None , resume_drop_last = True , save_after = True ,
                   indent : int = 0 , vb_level : int = 1):
        """Accounting portfolio through date, if cache is True, will cache the account"""
        if isinstance(config_or_benchmark , AccountConfig):
            config = config_or_benchmark
        else:
            config = AccountConfig(self.portfolio.name , AccountConfig.get_benchmark_name(config_or_benchmark) , start , end , analytic , attribution , 
                                   trade_engine = trade_engine , daily = daily , )
        self.benchmark = AccountConfig.get_benchmark(config_or_benchmark)
        self.account.config = config
        self.resume_path = resume_path

        if Proj.Conf.Model.TRAIN.resume_test_fmp_account:
            self.resumed_account = PortfolioAccount.load(self.resume_path)
            if resume_drop_last:
                last_model_date = self.resumed_account.max_model_date
                if len(before_last_model_dates := self.port_dates[self.port_dates <= last_model_date]) > 0:
                    last_model_date = int(before_last_model_dates.max())
                self.resumed_account.filter_dates(end = last_model_date - 1)

            if resume_end is not None:
                self.resumed_account.filter_dates(end = resume_end)

            if not self.resumed_account.empty:
                Logger.success(f'Load Account from {self.resume_path} at {CALENDAR.dates_str(self.resumed_account.model_date)}' , 
                            indent = indent + 1 , vb_level = Proj.vb.max)
        else:
            self.resumed_account = PortfolioAccount()
        self.go(cache = cache , indent = indent , vb_level = vb_level)
        self.account.with_index(with_index)
        if save_after:
            self.account.save(self.resume_path , indent = indent , vb_level = vb_level + 1)
        return self

    @property
    def config(self) -> AccountConfig:
        return self.account.config
    
    def go(self , cache = False , indent : int = 1 , vb_level : int = 2):
        if cache and self.config.key in self.cached_accounts:
            self.account = self.cached_accounts[self.config.key]
            return self

        if len(self.port_dates) == 0:
            model_dates = self.port_dates
        else:
            port_min , port_max = self.port_dates.min() , self.port_dates.max()
            start = np.max([port_min , self.config.start , self.resumed_account.max_model_date + 1])
            end   = np.min([DATAVENDOR.td(port_max,5) , self.config.end , DATAVENDOR.td(DATAVENDOR.last_quote_dt,-1)])
            model_dates = DATAVENDOR.td_within(start , end)

        if len(model_dates) == 0:
            self.account = self.resumed_account
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

        df = pd.DataFrame({
            'model_date':model_dates , 'start':period_st , 'end':period_ed ,
            'pf':0. , 'bm':0. , 'turn':0. , 'excess':0. , 'overnight': 0. ,
            'analytic':None , 'attribution':None}).set_index('model_date')

        port_old = Port.none_port(model_dates[0])
        Logger.stdout(f'{self.config.name} has {len(df)} account dates at {CALENDAR.dates_str([period_st[0] , period_ed[-1]])}' , 
                      indent = indent , vb_level = vb_level)
        for i , (mdate , ed) in enumerate(zip(model_dates , period_ed)):
            port_new = self.portfolio.get(mdate) if self.portfolio.has(mdate) else port_old
            bench = self.benchmark.get(mdate , True)

            rets = self.get_rets(port_old , port_new , bench , ed)
            turn = port_new.turnover(port_old)
            df.loc[mdate , ['pf' , 'bm' , 'overnight' , 'turn']] = \
                np.round([rets['pf'] , rets['bm'] , rets['overnight'] , turn] , Proj.Conf.Factor.ROUNDING.ret)
            
            if self.config.analytic: 
                df.loc[mdate , 'analytic']    = RISK_MODEL.get(mdate).analyze(port_new , bench , port_old) #type:ignore
            if self.config.attribution: 
                df.loc[mdate , 'attribution'] = RISK_MODEL.get(mdate).attribute(port_new , bench , ed , turn * self.config.trade_cost)  #type:ignore
            port_old = port_new.evolve_to_date(ed)
            if i > 0 and ((i + 1) % 100 == 0 or i == len(df) - 2):
                Logger.stdout(f'{self.config.name} accounting {i + 1} / {len(df) - 1} at {mdate}' , 
                              indent = indent , vb_level = vb_level + 2)

        df['pf']  = df['pf'] - df['turn'] * self.config.trade_cost
        df['excess'] = df['pf'] - df['bm']
        self.account.input = df.reset_index('model_date' , drop = False)
        self.account = PortfolioAccount.Concat(self.resumed_account , self.account)

        if cache:
            self.cached_accounts[self.config.key] = self.account
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

class PortfolioAccountManager:
    """
    Manage portfolio accounts in a directory.
    """
    def __init__(self , account_dir : str | Path):
        self.account_dir = Path(account_dir)
        self.accounts : dict[str , PortfolioAccount] = {}
        self.account_dir.mkdir(exist_ok=True)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.account_names})'
    
    def append_accounts(self , **accounts : PortfolioAccount):
        for name , new_account in accounts.items():
            self.accounts[name] = PortfolioAccount.Concat(self.accounts.get(name , None) , new_account)
        return self
    
    @property
    def account_names(self):
        return list(self.accounts.keys())
    
    def account_last_model_dates(self):
        return {name:account.max_model_date for name,account in self.accounts.items()}
    
    def account_last_end_dates(self):
        return {name:account.max_end_date for name,account in self.accounts.items()}

    def load_single(self , path : str | Path , missing_ok = True , append = True):
        path = Path(path)
        pkl_path = path.with_suffix('.pkl')
        tar_path = path.with_suffix('.tar')
        assert missing_ok or pkl_path.exists() or tar_path.exists() , f'{path} not exist'
        account = PortfolioAccount.load(tar_path if tar_path.exists() else pkl_path)
        if path.stem in self.accounts:
            if append:
                self.accounts[path.stem] = PortfolioAccount.Concat(self.accounts[path.stem] , account)
            else:
                raise KeyError(f'{path.stem} is already in the accounts')
        self.accounts[path.stem] = account
        return self
    
    def clear(self):
        self.accounts.clear()
        return self
    
    def load_dir(self , append = True):
        [self.load_single(path , append = append) for path in self.account_dir.iterdir() if path.suffix == '.pkl']
        return self
    
    def deploy(self , fmp_names : list[str] | None = None , overwrite = False , indent : int = 0 , vb_level : int = 1):
        if fmp_names is None: 
            fmp_names = list(self.accounts.keys())
        fmp_paths = {name:self.account_dir.joinpath(f'{name}.tar') for name in fmp_names}
        if not overwrite:
            existed = [path for path in fmp_paths.values() if path.exists()]
            assert not existed , f'Existed paths : {existed}'
        for name in fmp_names:
            self.accounts[name].save(fmp_paths[name] , indent = indent , vb_level = vb_level)
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
        account = PortfolioAccount.Total(*dfs.values())
        
        if account.empty: 
            return self
        task = self.select_analytic(category , task_name , **kwargs)
        task.calc(account)
        if plot: 
            task.plot(show = display)  
        return self      