import itertools , time
import numpy as np

from typing import Any , Literal

from src.basic import Timer

from ..util import Portfolio , Benchmark , AlphaModel , RISK_MODEL , PortCreateResult , PortfolioAccountant
from .optimizer import OptimizedPortfolioCreator
from .generator import TopStocksPortfolioCreator , ScreeningPortfolioCreator
from .fmp_basic import (get_prefix , get_port_index , get_strategy_name , get_suffix , 
                        get_full_name , get_benchmark , get_benchmark_name , parse_full_name)

class PortfolioBuilder:
    '''
    alpha : AlphaModel
    benchmark : Benchmark | Portfolio | Port | str
    category : Literal['optim' , 'top'] | Any
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
    '''
    def __init__(self , category : Literal['optim' , 'top' , 'screen'] | Any , 
                 alpha : AlphaModel , benchmark : Portfolio | Benchmark | str | None = None, lag : int = 0 ,
                 strategy : str = 'default' , suffixes : list[str] | str = [] , 
                 build_on : Portfolio | None = None , verbosity : int = 1 , **kwargs):
        self.category  = category
        self.alpha     = alpha
        self.benchmark = get_benchmark(benchmark)
        self.kwargs    = kwargs
        self.lag       = lag
        self.verbosity = verbosity
        
        self.prefix         = get_prefix(category)
        self.factor_name    = alpha.name
        self.benchmark_name = get_benchmark_name(benchmark)
        self.strategy       = get_strategy_name(category , strategy , kwargs)
        self.suffix         = get_suffix(lag , suffixes)

        self.portfolio = Portfolio(self.full_name) if build_on is None else build_on
        self.creations : list[PortCreateResult] = []

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name=\'{self.full_name}\',kwargs={self.kwargs},'+\
            f'{len(self.portfolio)} fmps,'+'not '* (not hasattr(self , 'account')) + 'accounted)'
    
    @property
    def full_name(self):
        return '.'.join([self.prefix , self.factor_name , self.benchmark.name , self.strategy , self.suffix])
    
    @property
    def port_index(self):
        return get_port_index(self.full_name)
    
    @classmethod
    def from_full_name(cls , full_name : str , alpha : AlphaModel , build_on : Portfolio | None = None , verbosity : int = 1 , **kwargs):
        elements = parse_full_name(full_name)
        assert alpha.name.lower() == elements['factor_name'].lower() , f'Alpha name mismatch: {alpha.name} != {elements["factor_name"]}'
        return cls(alpha = alpha , build_on = build_on , verbosity = verbosity , **elements , **kwargs)
    
    @staticmethod
    def get_full_name(category : Literal['optim' , 'top' , 'screen'] , alpha : AlphaModel | str , 
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
            case _:
                raise ValueError(f'Unknown category: {self.category}')

        self.creator = creator_class(self.full_name).setup(print_info = verbosity > 0 , **self.kwargs)
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

class PortfolioBuilderGroup:
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
                 category : Literal['optim' , 'top' , 'screen'] | Any ,
                 alpha_models : AlphaModel | list[AlphaModel] , 
                 benchmarks : str | None | list = None , 
                 add_lag : int = 0 , 
                 param_groups : dict[Any,dict[str,Any]] = {} ,
                 daily : bool = False ,
                 analytic : bool = True ,
                 attribution : bool = True ,
                 trade_engine : Literal['default' , 'harvest' , 'yale'] = 'default' ,
                 verbosity : int = 1 ,
                 **kwargs):
        self.builders : list[PortfolioBuilder] = []
        self.category = category

        assert alpha_models , f'alpha_models must has elements!'
        self.alpha_models = alpha_models if isinstance(alpha_models , list) else [alpha_models]
        self.relevant_dates = np.unique(np.concatenate([amodel.available_dates() for amodel in self.alpha_models]))
        self.benchmarks = Benchmark.get_benchmarks(benchmarks)

        assert add_lag >= 0 , add_lag
        self.lags = [0 , add_lag] if add_lag > 0 else [0]

        if param_groups:
            self.param_groups = {key:(kwargs | kwg) for key,kwg in param_groups.items()}
        else:
            self.param_groups = {'default':kwargs}
        self.acc_kwargs : dict[str,Any] = {'daily' : daily , 'analytic' : analytic , 'attribution' : attribution , 'trade_engine' : trade_engine}

        self.verbosity = verbosity

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.category_title} , {len(self.alpha_models)} alphas , {len(self.benchmarks)} benchmarks , ' + \
               f'{len(self.lags)} lags , {len(self.param_groups)} param_groups , {len(self.relevant_dates)} dates , ' + \
               f'({len(self.alpha_models) * len(self.benchmarks) * len(self.lags) * len(self.param_groups) * len(self.relevant_dates)} builds)'
    
    def setup_builders(self):
        self.builders.clear()
        for (alpha , lag , bench , strategy) in itertools.product(self.alpha_models , self.lags , self.benchmarks , self.param_groups):
            kwargs = self.param_groups[strategy] | {'strategy':strategy , 'verbosity':self.verbosity - 1}
            builder = PortfolioBuilder(self.category , alpha , bench , lag , **kwargs).setup()
            self.builders.append(builder)
        return self
    
    @property
    def category_title(self):
        if self.category == 'optim':
            return 'Optimized'
        elif self.category == 'top':
            return 'TopStocks'
        elif self.category == 'screen':
            return 'Screening'
        else:
            return self.category.title()
    
    @property
    def port_name_nchar(self):
        if not hasattr(self , '_port_name_nchar'):
            self._port_name_nchar = np.max([len(builder.full_name) for builder in self.builders])
        return self._port_name_nchar

    def building(self):
        RISK_MODEL.load_models(self.relevant_dates)
        self.setup_builders()
        self.print_in_optimization('start')
        for self._date in self.relevant_dates:
            for self._builder in self.builders:
                if not self._builder.alpha.has(self._date): 
                    continue
                self._builder.build(self._date)
                self.print_in_optimization('loop')
        self.print_in_optimization('end')
        return self
    
    def accounting(self , start : int = -1 , end : int = 99991231):
        with Timer(f'{self.__class__.__name__} accounting' , silent = self.verbosity < 1):
            for builder in self.builders:
                with Timer(f'{builder.portfolio.name} accounting' , silent = self.verbosity < 2):
                    builder.accounting(start , end , **self.acc_kwargs)
        return self
    
    def total_account(self):
        assert self.builders , 'No builders to account!'
        df = PortfolioAccountant.total_account([builder.account for builder in self.builders])
        return df
    
    def print_in_optimization(self , where : Literal['start' , 'loop' , 'end']):
        if self.verbosity > 0:
            if where == 'start':
                self.t0 = time.time()
                print(f'{str(self)} start!')
                self.opt_count = 0
            elif where == 'loop':
                if self.verbosity > 1 and self.opt_count % 100 == 0: 
                    time_cost = {k:float(np.round(v*1000,2)) for k,v in self._builder.creations[-1].time.items()}
                    print(f'building of {self.opt_count:4d}th [{self._builder.portfolio.name:{self.port_name_nchar}s}]' + 
                          f' Finished at {self._date} , time cost (ms) : {time_cost}')
                self.opt_count += 1
            elif where == 'end':
                self.t1 = time.time()
                total_time = f'cost {self.t1-self.t0:.2f} secs'
                each_time = f'{(self.t1-self.t0)/max(self.opt_count,1)*1000:.1f} ms per building'
                print(f'{self.__class__.__name__} building finished, {total_time}, {each_time}')
