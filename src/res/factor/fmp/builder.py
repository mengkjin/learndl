import itertools
import numpy as np

from datetime import datetime
from pathlib import Path
from typing import Any , Literal

from src.proj import Duration , Logger , CALENDAR , Proj

from ..util import Portfolio , Benchmark , AlphaModel , RISK_MODEL , PortCreateResult , PortfolioAccountant
from .optimizer import OptimizedPortfolioCreator
from .generator import TopStocksPortfolioCreator , ScreeningPortfolioCreator , RevScreeningPortfolioCreator
from .fmp_basic import (get_prefix , get_port_index , get_strategy_name , get_suffix , get_factor_name ,
                        get_full_name , get_benchmark , get_benchmark_name , parse_full_name)

class PortfolioBuilder:
    '''
    alpha : AlphaModel
    benchmark : Benchmark | Portfolio | Port | str
    category : Literal['optim' , 'top' , 'screen'] | Any
    lag : int , lag periods (not days)
    strategy : str
    suffixes : list[str] | str
    build_on : Portfolio | None

    optim accepted kwargs:
        prob_type : PROB_TYPE = 'quadprog'
        engine_type : ENGINE_TYPE = 'mosek'
        cvxpy_solver : CVXPY_SOLVER = 'mosek'
        config_path : str | None = None
        opt_relax : bool = True
        opt_turn  : bool = True
        opt_qobj  : bool = True
        opt_qcon  : bool = True
        opt_short : bool = True
    top accepted kwargs:
        n_best : int = 50
        turn_control : float = 0.2
        buffer_zone : float = 0.8
        no_zone : float = 0.5
        indus_control : float = 0.1
    screen accepted kwargs:
        screen_ratio : float = 0.5
        sorting_alpha : tuple[str , str , str | None] = ('pred' , 'gru_day_V1' , None)
        n_best : int = 50
        turn_control : float = 0.2
        buffer_zone : float = 0.8
        no_zone : float = 0.5
        indus_control : float = 0.1
    revscreen accepted kwargs:
        screen_ratio : float = 0.5
        screen_alpha : tuple[str , str , str | None] = ('pred' , 'gru_day_V1' , None)
        n_best : int = 50
        turn_control : float = 0.2
        buffer_zone : float = 0.8
        no_zone : float = 0.5
        indus_control : float = 0.1
    '''
    def __init__(self , category : Literal['optim' , 'top' , 'screen' , 'revscreen'] | Any , 
                 alpha : AlphaModel , benchmark : Portfolio | Benchmark | str | None = None, lag : int = 0 ,
                 strategy : str = 'default' , suffixes : list[str] | str = [] , 
                 build_on : Portfolio | None = None , resume_path : Path | str | None = None , 
                 indent : int = 0 , vb_level : int = 1 , **kwargs):

        assert build_on is None or resume_path is None , 'build_on and resume_path cannot be provided together'
        self.category     = category
        self.alpha        = alpha
        self.benchmark    = get_benchmark(benchmark)
        self.kwargs       = kwargs
        self.lag          = lag
        self.build_on     = build_on 
        self.resume_path  = resume_path
        self.indent       = indent
        self.vb_level     = vb_level
        
        self.prefix         = get_prefix(category)
        self.factor_name    = get_factor_name(alpha)
        self.benchmark_name = get_benchmark_name(benchmark)
        self.strategy       = get_strategy_name(category , strategy , kwargs)
        self.suffix         = get_suffix(lag , suffixes)

        self.creations : list[PortCreateResult] = []

        self.set_build_on(self.build_on)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name=\'{self.full_name}\',kwargs={self.kwargs},'+\
            f'{len(self.portfolio)} fmps,'+'not '* (not hasattr(self , 'account')) + 'accounted)'

    @property
    def min_alpha_date(self) -> int:
        dates = self.alpha.available_dates()
        return dates.min() if len(dates) > 0 else 99991231

    @property
    def resumed_portfolio_end_date(self) -> int:
        if not hasattr(self , '_resumed_portfolio_end_date'):
            self._resumed_portfolio_end_date = -1
        return self._resumed_portfolio_end_date

    @resumed_portfolio_end_date.setter
    def resumed_portfolio_end_date(self , value : int):
        self._resumed_portfolio_end_date = value

    @property
    def resumed_account_end_date(self) -> int:
        return min(int(self.resumed_account['model_date'].iloc[-2]) if len(self.resumed_account) > 1 else -1 , self.resumed_portfolio_end_date)

    @property
    def resume_path_portfolio(self):
        if self.resume_path is None:
            return None
        else:
            return Path(self.resume_path) / 'portfolio' / f'{self.full_name.lower()}.feather'

    @property
    def resume_path_account(self):
        if self.resume_path is None:
            return None
        else:
            return Path(self.resume_path) / 'account' / f'{self.full_name.lower()}'

    def set_build_on(self , build_on : Portfolio | None = None , start : int = -1 , end : int = 99991231):
        if build_on is None:
            self.portfolio = Portfolio(self.full_name)
        else:
            dates = build_on.port_date[(build_on.port_date >= start) & (build_on.port_date <= end) & (build_on.port_date < self.min_alpha_date)]
            self.portfolio = build_on.filter_dates(dates = dates).rename(self.full_name)
            self.resumed_portfolio_end_date = -1 if self.portfolio.is_empty else self.portfolio.port_date.max()
        return self

    def load_portfolio(self , start : int = -1 , end : int = 99991231):
        if self.resume_path_portfolio is not None:
            port = Portfolio.load(self.resume_path_portfolio)
            dates = port.port_date[(port.port_date >= start) & (port.port_date <= end) & (port.port_date < self.min_alpha_date)]
            self.portfolio = port.filter_dates(dates = dates).rename(self.full_name)
            self.resumed_portfolio_end_date = -1 if self.portfolio.is_empty else self.portfolio.port_date.max()
            Logger.success(f'Load portfolio from {self.resume_path_portfolio} at {CALENDAR.dates_str(self.portfolio.port_date)}' , 
                           indent = self.indent , vb_level = Proj.vb_max)
        return self

    def save_portfolio(self , append = False):
        if self.resume_path_portfolio is None:
            return self
        self.portfolio.save(self.resume_path_portfolio , overwrite = True , append = append , indent = self.indent , vb_level = self.vb_level + 1)
        return self
    
    @property
    def full_name(self):
        return '.'.join([self.prefix , self.factor_name , self.benchmark_name , self.strategy , self.suffix])
    
    @property
    def port_index(self):
        return get_port_index(self.full_name)
    
    def setup(self):
        match self.category:
            case 'optim':
                creator_class = OptimizedPortfolioCreator
            case 'top':
                creator_class = TopStocksPortfolioCreator
            case 'screen':
                creator_class = ScreeningPortfolioCreator
            case 'revscreen':
                creator_class = RevScreeningPortfolioCreator
            case _:
                raise ValueError(f'Unknown category: {self.category}')

        self.creator = creator_class(self.full_name).setup(indent = self.indent , vb_level = self.vb_level + 1 , **self.kwargs)
        return self
        
    def build(self , date : int):
        assert hasattr(self , 'creator') , 'PortfolioBuilder not setup!'
        assert self.alpha.has(date) , f'{self.alpha.name} has no data at {date}'
        init_port = self.portfolio.get(date , latest = True)
        port_rslt = self.creator.create(date , self.alpha.get(date , lag = self.lag) , self.benchmark , init_port)
        self.creations.append(port_rslt)
        self.portfolio.append(port_rslt.port.with_name(self.portfolio.name))
        self.port = port_rslt.port
        return self

    def load_account(self):
        self.resumed_account = PortfolioAccountant.load(self.resume_path_account)
        if not self.resumed_account.empty:
            Logger.success(f'Load account from {self.resume_path_account} at {CALENDAR.dates_str(self.resumed_account['model_date'])}' , 
                           indent = self.indent , vb_level = Proj.vb_max)
        return self
    
    def accounting(self , start : int = -1 , end : int = 99991231 ,
                   analytic = True , attribution = True , * ,
                   trade_engine : Literal['default' , 'harvest' , 'yale'] = 'default' ,
                   daily = False):
        '''Accounting portfolio through date, require at least portfolio'''
        self.portfolio.accounting(self.benchmark , max(start , self.resumed_account_end_date + 1) , end , 
                                  analytic and self.lag == 0 , attribution and self.lag == 0 ,
                                  trade_engine = trade_engine , daily = daily , indent = self.indent , vb_level = self.vb_level)
        if hasattr(self , 'resumed_account') and not self.resumed_account.empty:
            self.account = PortfolioAccountant.concat_accounts([self.resumed_account , self.portfolio.account])
        else:
            self.account = self.portfolio.account
        return self

    def save_account(self):
        PortfolioAccountant.save_account(self.account , self.resume_path_account , indent = self.indent , vb_level = self.vb_level + 1)
        return self

    def account_with_index(self):
        return PortfolioAccountant.account_with_index(self.account , self.port_index)

    @classmethod
    def from_full_name(cls , full_name : str , alpha : AlphaModel , build_on : Portfolio | None = None , indent : int = 0 , vb_level : int = 1 , **kwargs):
        elements = parse_full_name(full_name)
        assert alpha.name.lower() == elements['factor_name'].lower() , f'Alpha name mismatch: {alpha.name} != {elements["factor_name"]}'
        return cls(alpha = alpha , build_on = build_on , indent = indent , vb_level = vb_level , **elements , **kwargs)
    
    @staticmethod
    def get_full_name(category : Literal['optim' , 'top' , 'screen' , 'revscreen'] , alpha : AlphaModel | str , 
                      benchmark : Portfolio | Benchmark | str | None = None , 
                      strategy : str = 'default' , suffixes : list[str] | str = [] , lag : int = 0 , **kwargs):
        return get_full_name(category , alpha , benchmark , strategy , suffixes , lag , **kwargs)

class PortfolioGroupBuilder:
    '''
    parallel_kwargs:
        can have list of builder_kwargs' components, but cannot overlap with builder_kwargs
    builder_kwargs:
        optim accepted kwargs:
            prob_type : PROB_TYPE = 'quadprog'
            engine_type : ENGINE_TYPE = 'mosek'
            cvxpy_solver : CVXPY_SOLVER = 'mosek'
            config_path : str | None = None
            opt_relax : bool = True
            opt_turn  : bool = True
            opt_qobj  : bool = True
            opt_qcon  : bool = True
            opt_short : bool = True
        top accepted kwargs:
            n_best : int = 50
            turn_control : float = 0.2
            buffer_zone : float = 0.8
            no_zone : float = 0.5
            indus_control : float = 0.1
        screen accepted kwargs:
            screen_ratio : float = 0.5
            sorting_alpha : tuple[str , str , str | None] = ('pred' , 'gru_day_V1' , None)
            n_best : int = 50
            turn_control : float = 0.2
            buffer_zone : float = 0.8
            no_zone : float = 0.5
            indus_control : float = 0.1
    acc_kwargs:
        daily : bool = False
        analytic : bool = True
        attribution : bool = True
    '''
    def __init__(
        self , 
        category : Literal['optim' , 'top' , 'screen' , 'revscreen'] | Any ,
        alpha_models : AlphaModel | list[AlphaModel] , 
        benchmarks : str | None | list = None , 
        add_lag : int = 0 , 
        param_groups : dict[Any,dict[str,Any]] = {} ,
        daily : bool = False ,
        analytic : bool = True ,
        attribution : bool = True ,
        trade_engine : Literal['default' , 'harvest' , 'yale'] = 'default' ,
        resume : bool = False ,
        resume_path : Path | str | None = None ,
        start_dt : int = -1 ,
        end_dt : int = 99991231 ,
        caller = None ,
        indent : int = 0 , 
        vb_level : int = 1 ,
        **kwargs
    ):
        
        self.category = category

        assert alpha_models , f'alpha_models must has elements!'
        self.alpha_models = alpha_models if isinstance(alpha_models , list) else [alpha_models]
        self.relevant_dates = (np.unique(np.concatenate([amodel.available_dates() for amodel in self.alpha_models])) if self.alpha_models else np.array([] , dtype=int))
        self.relevant_dates = self.relevant_dates[(self.relevant_dates >= start_dt) & (self.relevant_dates <= end_dt)]
        self.benchmarks = Benchmark.get_benchmarks(benchmarks)

        assert add_lag >= 0 , add_lag
        self.lags = [0 , add_lag] if add_lag > 0 else [0]

        if param_groups:
            self.param_groups = {key:(kwargs | kwg) for key,kwg in param_groups.items()}
        else:
            self.param_groups = {'default':kwargs}
        self.acc_kwargs : dict = {
            'analytic' : analytic , 
            'attribution' : attribution , 
            'daily' : daily , 
            'trade_engine' : trade_engine}

        self.indent = indent
        self.vb_level = vb_level
        self.resume = resume
        self.resume_path = Path(resume_path) if resume_path is not None and resume else None
        self.caller = caller
        self.start_dt = start_dt
        self.end_dt = end_dt

        self.builders : list[PortfolioBuilder] = []
        self.accounted = False

    @property
    def n_builders(self):
        return len(self.alpha_models) * len(self.benchmarks) * len(self.lags) * len(self.param_groups)

    @property
    def n_builds(self):
        return self.n_builders * len(self.relevant_dates)

    @property
    def class_name(self):
        if self.caller is None:
            return self.__class__.__name__
        else:
            return f'{str(self.caller)}.{self.__class__.__name__}'

    def __repr__(self) -> str:
        return f'{self.class_name}({self.category_title}, {self.n_builders} builders[{len(self.alpha_models)}alpha,{len(self.benchmarks)}bm,' + \
               f'{len(self.lags)}lag,{len(self.param_groups)}kwgs] & {len(self.relevant_dates)} dates, totaling {self.n_builds} builds)'

    def builders_info(self):
        Logger.stdout(f'{self.class_name} has {self.n_builders} builders ({len(self.alpha_models)} alphas x {len(self.benchmarks)} bms x {len(self.lags)} lags x {len(self.param_groups)} kwgs) {self.n_builds} builds (x {len(self.relevant_dates)} dates)' , 
                      indent = self.indent , vb_level = self.vb_level)
    
    def builders_setup(self):
        with Logger.Timer(f'{self.class_name}.setup' , indent = self.indent , vb_level = self.vb_level , enter_vb_level = self.vb_level + 2):
            self.builders.clear()
            self.accounted = False
            for (alpha , lag , bench , strategy) in itertools.product(self.alpha_models , self.lags , self.benchmarks , self.param_groups):
                kwargs = self.param_groups[strategy] | {'strategy':strategy , 'indent':self.indent + 1 , 'vb_level':self.vb_level + 1 , 'resume_path':self.resume_path}
                builder = PortfolioBuilder(self.category , alpha , bench , lag , **kwargs).setup()
                self.builders.append(builder)
        return self

    def builders_resume(self):
        if self.resume_path is None:
            return
        with Logger.Timer(f'{self.class_name}.resume' , indent = self.indent , vb_level = self.vb_level):
            for builder in self.builders:
                builder.load_portfolio(start = self.start_dt , end = self.end_dt)
        return self

    def save_portfolios(self):
        if self.resume_path is None:
            return
        with Logger.Timer(f'{self.class_name}.export' , indent = self.indent , vb_level = self.vb_level , enter_vb_level = self.vb_level + 2):
            for builder in self.builders:
                builder.save_portfolio(append = True)
        return self
    
    @property
    def category_title(self):
        if self.category == 'optim':
            return 'Optimized'
        elif self.category == 'top':
            return 'TopStocks'
        elif self.category == 'screen':
            return 'Screening'
        elif self.category == 'revscreen':
            return 'RevScreening'
        else:
            return self.category.title()
    
    @property
    def port_name_nchar(self):
        if not hasattr(self , '_port_name_nchar'):
            self._port_name_nchar = np.max([len(builder.full_name) for builder in self.builders])
        return self._port_name_nchar

    def build(self):
        self.builders_info()
        RISK_MODEL.load_models(self.relevant_dates)
        
        self.builders_setup()
        self.builders_resume()

        _t0 = datetime.now()
        _opt_count = 0
            
        for self._date in self.relevant_dates:
            for self._builder in self.builders:
                if not self._builder.alpha.has(self._date): 
                    continue
                self._builder.build(self._date)
                _opt_count += 1
                if _opt_count % 100 == 0: 
                    time_cost = {k:float(np.round(v*1000,2)) for k,v in self._builder.creations[-1].timecost.items()}
                    Logger.stdout(f'Building of {_opt_count:4d}th [{self._builder.portfolio.name:{self.port_name_nchar}s}]' + 
                          f' Finished at {self._date} , time cost (ms) : {time_cost}' , indent = self.indent + 1 , vb_level = self.vb_level + 1)
        secs = (datetime.now() - _t0).total_seconds()
        Logger.stdout(f'{self.class_name}.build finished! Cost {Duration(secs)}, {(secs/max(_opt_count,1)*1000):.1f} ms per building' , 
                      indent = self.indent , vb_level = self.vb_level)
        self.save_portfolios()
        return self
    
    def accounting(self):
        with (
            Logger.Timer(f'{self.class_name} accounting' , indent = self.indent , vb_level = self.vb_level , enter_vb_level = self.vb_level + 1) , 
            Logger.Profiler(False , output = f'{self.caller.__class__.__name__}_{self.__class__.__name__}_accounting.csv')
        ):
            for builder in self.builders:
                with Logger.Timer(f'{builder.portfolio.name} accounting' , indent = self.indent + 1 , vb_level = self.vb_level + 1):
                    builder.load_account().accounting(self.start_dt , self.end_dt , **self.acc_kwargs).save_account()
        self.accounted = True
        return self
    
    def accounts(self):
        assert self.builders , 'No builders to account!'
        if not self.accounted:
            self.accounting()
        df = PortfolioAccountant.total_account([builder.account_with_index() for builder in self.builders])
        return df