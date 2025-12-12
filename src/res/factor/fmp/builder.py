import itertools
import numpy as np

from datetime import datetime
from pathlib import Path
from typing import Any , Literal

from src.proj import Timer , Duration

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
                 build_on : Portfolio | None = None , overwrite : bool = False , verbosity : int = 1 , **kwargs):
        self.category  = category
        self.alpha     = alpha
        self.benchmark = get_benchmark(benchmark)
        self.kwargs    = kwargs
        self.lag       = lag
        self.verbosity = verbosity
        
        self.prefix         = get_prefix(category)
        self.factor_name    = get_factor_name(alpha)
        self.benchmark_name = get_benchmark_name(benchmark)
        self.strategy       = get_strategy_name(category , strategy , kwargs)
        self.suffix         = get_suffix(lag , suffixes)

        self.creations : list[PortCreateResult] = []

        self.set_build_on(build_on , overwrite)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name=\'{self.full_name}\',kwargs={self.kwargs},'+\
            f'{len(self.portfolio)} fmps,'+'not '* (not hasattr(self , 'account')) + 'accounted)'

    def build_on_path(self , path : Path | str):
        path = Path(path)
        if path.exists():
            if path.is_dir():
                return path.joinpath(f'{self.full_name.lower()}.feather')
            else:
                assert path.suffix == '.feather' , f'{path} is not a feather file'
                return path
        else:
            if path.suffix == '.feather':
                return path
            else:
                return path.joinpath(f'{self.full_name.lower()}.feather')

    def set_build_on(self , build_on : Portfolio | Path | str | None , overwrite : bool = False):
        if build_on is None:
            self.portfolio = Portfolio(self.full_name)
            return self
        if isinstance(build_on , Portfolio):
            port = build_on
        elif isinstance(build_on , (Path , str)):
            port = Portfolio.load(self.build_on_path(build_on))
        else:
            raise ValueError(f'Unknown build_on type: {type(build_on)}')
        port = port.rename(self.full_name)
        if not overwrite and port is not None and not port.is_empty:
            port = port.filter_dates(dates = port.port_date[port.port_date < self.alpha.available_dates().min()] , inplace = True)
        self.portfolio = port
        return self

    def export_portfolio(self , path : Path | str | None = None):
        if path is None:
            return
        path = self.build_on_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.portfolio.save(path , overwrite = True)
        return self
    
    @property
    def full_name(self):
        return '.'.join([self.prefix , self.factor_name , self.benchmark_name , self.strategy , self.suffix])
    
    @property
    def port_index(self):
        return get_port_index(self.full_name)
    
    @classmethod
    def from_full_name(cls , full_name : str , alpha : AlphaModel , build_on : Portfolio | None = None , verbosity : int = 1 , **kwargs):
        elements = parse_full_name(full_name)
        assert alpha.name.lower() == elements['factor_name'].lower() , f'Alpha name mismatch: {alpha.name} != {elements["factor_name"]}'
        return cls(alpha = alpha , verbosity = verbosity , **elements , **kwargs).set_build_on(build_on , False)
    
    @staticmethod
    def get_full_name(category : Literal['optim' , 'top' , 'screen' , 'revscreen'] , alpha : AlphaModel | str , 
                      benchmark : Portfolio | Benchmark | str | None = None , 
                      strategy : str = 'default' , suffixes : list[str] | str = [] , lag : int = 0 , **kwargs):
        return get_full_name(category , alpha , benchmark , strategy , suffixes , lag , **kwargs)
    
    def setup(self , verbosity : int | None = None):
        if verbosity is None: 
            verbosity = self.verbosity

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

        self.creator = creator_class(self.full_name).setup(print_info = verbosity > 1 , **self.kwargs)
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
    
    def accounting(self , start : int = -1 , end : int = 99991231 ,
                   analytic = True , attribution = True ,
                   trade_engine : Literal['default' , 'harvest' , 'yale'] = 'default' ,
                   daily = False):
        '''Accounting portfolio through date, require at least portfolio'''
        self.portfolio.accounting(self.benchmark , start , end , 
                                  analytic and self.lag == 0 , attribution and self.lag == 0 ,
                                  trade_engine , daily)
        self.account = self.portfolio.account_with_index(self.port_index)
        return self

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
    def __init__(self , 
                 category : Literal['optim' , 'top' , 'screen' , 'revscreen'] | Any ,
                 alpha_models : AlphaModel | list[AlphaModel] , 
                 benchmarks : str | None | list = None , 
                 add_lag : int = 0 , 
                 param_groups : dict[Any,dict[str,Any]] = {} ,
                 daily : bool = False ,
                 analytic : bool = True ,
                 attribution : bool = True ,
                 trade_engine : Literal['default' , 'harvest' , 'yale'] = 'default' ,
                 verbosity : int = 1 ,
                 resume : bool = False ,
                 resume_path : Path | str | None = None ,
                 caller = None ,
                 **kwargs):
        
        self.category = category

        assert alpha_models , f'alpha_models must has elements!'
        self.alpha_models = alpha_models if isinstance(alpha_models , list) else [alpha_models]
        self.relevant_dates = np.unique(np.concatenate([amodel.available_dates() for amodel in self.alpha_models])).astype(int)
        self.benchmarks = Benchmark.get_benchmarks(benchmarks)

        assert add_lag >= 0 , add_lag
        self.lags = [0 , add_lag] if add_lag > 0 else [0]

        if param_groups:
            self.param_groups = {key:(kwargs | kwg) for key,kwg in param_groups.items()}
        else:
            self.param_groups = {'default':kwargs}
        self.acc_kwargs : dict[str,Any] = {'daily' : daily , 'analytic' : analytic , 'attribution' : attribution , 'trade_engine' : trade_engine}

        self.verbosity = verbosity
        self.resume = resume
        self.resume_path = Path(resume_path) if resume_path is not None else None
        self.caller = caller

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
        print(f'{self.class_name} has {self.n_builders} builders ({len(self.alpha_models)} alphas x {len(self.benchmarks)} bms x {len(self.lags)} lags x {len(self.param_groups)} kwgs) and {self.n_builds} (x {len(self.relevant_dates)} dates)')\
    
    def builders_setup(self):
        with Timer(f'{self.class_name}.setup' , silent = self.verbosity < 1):
            self.builders.clear()
            self.accounted = False
            for (alpha , lag , bench , strategy) in itertools.product(self.alpha_models , self.lags , self.benchmarks , self.param_groups):
                kwargs = self.param_groups[strategy] | {'strategy':strategy , 'verbosity':self.verbosity - 1}
                builder = PortfolioBuilder(self.category , alpha , bench , lag , **kwargs).setup()
                self.builders.append(builder)
        return self

    def builders_resume(self):
        if self.resume_path is None or not self.resume:
            return
        with Timer(f'{self.class_name}.resume' , silent = self.verbosity < 1):
            for builder in self.builders:
                builder.set_build_on(self.resume_path)
        return self

    def export_portfolio(self):
        if self.resume_path is None:
            return
        with Timer(f'{self.class_name}.export' , silent = self.verbosity < 1):
            for builder in self.builders:
                builder.export_portfolio(self.resume_path)
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
        if self.verbosity > 1:
            print(f'{self.class_name}.building start')
            
        for self._date in self.relevant_dates:
            for self._builder in self.builders:
                if not self._builder.alpha.has(self._date): 
                    continue
                self._builder.build(self._date)
                _opt_count += 1
                if self.verbosity > 2 and _opt_count % 100 == 0: 
                    time_cost = {k:float(np.round(v*1000,2)) for k,v in self._builder.creations[-1].timecost.items()}
                    print(f'building of {_opt_count:4d}th [{self._builder.portfolio.name:{self.port_name_nchar}s}]' + 
                          f' Finished at {self._date} , time cost (ms) : {time_cost}')
        if self.verbosity > 0:
            secs = (datetime.now() - _t0).total_seconds()
            print(f'{self.class_name}.build finished! Cost {Duration(secs)}, {(secs/max(_opt_count,1)*1000):.1f} ms per building')
        self.export_portfolio()
        return self
    
    def accounting(self , start : int = -1 , end : int = 99991231):
        with Timer(f'{self.class_name} accounting' , silent = self.verbosity < 1):
            for builder in self.builders:
                with Timer(f'  --> {builder.portfolio.name} accounting' , silent = self.verbosity < 2):
                    builder.accounting(start , end , **self.acc_kwargs)
        self.accounted = True
        return self
    
    def accounts(self):
        assert self.builders , 'No builders to account!'
        if not self.accounted:
            self.accounting()
        df = PortfolioAccountant.total_account([builder.account for builder in self.builders])
        return df