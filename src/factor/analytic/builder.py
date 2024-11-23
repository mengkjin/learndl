import itertools , time
import pandas as pd
import numpy as np
import torch

from contextlib import nullcontext
from typing import Any , Literal , Optional

from src.basic import CONF ,Timer
from src.basic.conf import ROUNDING_RETURN , ROUNDING_TURNOVER , TRADE_COST
from src.data import DATAVENDOR
from src.factor.util import Portfolio , Benchmark , AlphaModel , RISK_MODEL , PortfolioOptimizer , PortCreateResult , Port , PortfolioGenerator

class PortfolioBuilder:
    '''
    optim accepted kwargs:
        prob_type : PROB_TYPE = 'quadprog'
        engine_type : ENGINE_TYPE = 'mosek'
        cvxpy_solver : CVXPY_SOLVER = 'mosek'
        config_path : Optional[str] = None
        opt_relax : bool = True
        opt_turn  : bool = True
        opt_qobj  : bool = True
        opt_qcon  : bool = True
        opt_short : bool = True
    top accepted kwargs:
        n_best : int = 50
        turn_control : float = 0.2
        buffer_zone : float = 0.8
        indus_control : float = 0.1
    '''
    def __init__(self , category : Literal['optim' , 'top'] | Any , 
                 alpha : AlphaModel , benchmark : Optional[Portfolio | Benchmark] = None, lag : int = 0 ,
                 strategy : str = 'default' , build_on : Optional[Portfolio] = None , verbosity : int = 1 , **kwargs):
        self.category = category
        self.alpha = alpha
        self.benchmark = Portfolio(is_default=True) if benchmark is None else benchmark
        self.lag = lag
        self.verbosity = verbosity

        if strategy and strategy != 'default':
            self.strategy = strategy
        else:
            if self.category == 'top':
                n = kwargs['n_best'] if 'n_best' in kwargs else PortfolioGenerator.DEFAULT_N_BEST
                self.strategy = f'Top{n:_>3d}'
            else:
                self.strategy = self.category.capitalize()

        self.kwargs = kwargs

        self.portfolio = Portfolio(self.full_name()) if build_on is None else build_on
        self.creations : list[PortCreateResult] = []

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(category=\'{self.category}\',name=\'{self.full_name()}\',alpha=\'{self.alpha.name}\','+\
            f'benchmark=\'{self.benchmark.name}\',lag={self.lag},kwargs={self.kwargs},'+\
            f'{len(self.portfolio)} fmp\'s,'+'not '* (self.account is None) + 'accounted)'

    def prefix(self): return 'Fmp'
        
    def suffix(self): return f'lag{self.lag}'
    
    def full_name(self):
        return f'{self.prefix()}.{self.alpha.name}.{self.benchmark.name}.{self.strategy}.{self.suffix()}'
    
    def port_index(self):
        default_index : dict[str,Any] = {
            'prefix'      : self.prefix() ,
            'factor_name' : self.alpha.name ,
            'benchmark'   : self.benchmark.name ,
            'strategy'    : self.strategy ,
            'suffix'      : self.suffix() ,
        }

        default_index['lag'] = self.lag
        if self.category == 'top' and 'n_best' in self.kwargs:
            default_index['topN'] = self.kwargs['n_best']
        return default_index
    
    def setup(self):
        if self.category == 'optim':
            self.creator = PortfolioOptimizer(self.full_name()).setup(print_info = self.verbosity > 0 , **self.kwargs)
        elif self.category == 'top':
            self.creator = PortfolioGenerator(self.full_name()).setup(print_info = self.verbosity > 0 , **self.kwargs)
        else:
            raise ValueError(f'Unknown category: {self.category}')
        return self
        
    def build(self , date : int):
        assert self.alpha.has(date) , f'{self.alpha.name} has no data at {date}'
        init_port = self.portfolio.get(date , latest = True)
        port_rslt = self.creator.create(date , self.alpha.get(date , lag = self.lag) , self.benchmark , init_port)
        self.creations.append(port_rslt)
        self.portfolio.append(port_rslt.port)
        return self
    
    def accounting(self , start : int = -1 , end : int = 99991231 , daily = False , 
                   analytic = True , attribution = True):
        '''Accounting portfolio through date, require at least portfolio'''
        port_min , port_max = self.portfolio.available_dates().min() , self.portfolio.available_dates().max()
        start = np.max([port_min , start])
        end   = np.min([DATAVENDOR.td(port_max,5).td , end , DATAVENDOR.td(DATAVENDOR.last_quote_dt,-1).td])

        model_dates = DATAVENDOR.td_within(start , end)
        if daily:
            period_st = DATAVENDOR.td_array(model_dates , 1)
            period_ed = period_st
        else:
            model_dates = np.intersect1d(model_dates , self.portfolio.available_dates())
            period_st = DATAVENDOR.td_array(model_dates , 1)
            period_ed = np.concatenate([model_dates[1:] , [DATAVENDOR.td(end,1).td]])

        assert np.all(model_dates < period_st) , (model_dates , period_st)
        assert np.all(period_st <= period_ed) , (period_st , period_ed)

        index = self.port_index()
        account = pd.DataFrame(index | {
            'model_date':np.concatenate([[-1],model_dates]) , 
            'start':np.concatenate([[model_dates[0]],period_st]) , 
            'end':np.concatenate([[model_dates[0]],period_ed]) ,
            'pf':0. , 'bm':0. , 'turn':0. , 'excess':0. ,
            'analytic':None , 'attribution':None}).set_index('model_date').sort_index()

        port_old = Port.none_port(model_dates[0])
        for date , ed in zip(model_dates , period_ed):
            port_new = self.portfolio.get(date) if self.portfolio.has(date) else port_old
            bench = self.benchmark.get(date , True)

            turn = np.round(port_new.turnover(port_old),ROUNDING_TURNOVER)
            account.loc[date , ['pf' , 'bm' , 'turn']] = \
                [np.round(port_new.fut_ret(ed) , ROUNDING_RETURN) , np.round(bench.fut_ret(ed) , ROUNDING_RETURN) , turn]
            
            if analytic and self.lag == 0: 
                account.loc[date , 'analytic']    = RISK_MODEL.get(date).analyze(port_new , bench , port_old) #type:ignore
            if attribution and self.lag == 0: 
                account.loc[date , 'attribution'] = RISK_MODEL.get(date).attribute(port_new , bench , ed , turn * TRADE_COST)  #type:ignore
            port_old = port_new.evolve_to_date(ed)

        account['pf']  = account['pf'] - account['turn'] * TRADE_COST
        account['excess'] = account['pf'] - account['bm']
        self.account = account.reset_index().set_index(list(index.keys())).sort_values('model_date')
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
            config_path : Optional[str] = None
            opt_relax : bool = True
            opt_turn  : bool = True
            opt_qobj  : bool = True
            opt_qcon  : bool = True
            opt_short : bool = True
        top accepted kwargs:
            n_best : int = 50
            turn_control : float = 0.2
            buffer_zone : float = 0.8
            indus_control : float = 0.1
    acc_kwargs:
        daily : bool = False
        analytic : bool = True
        attribution : bool = True
    '''
    def __init__(self , 
                 category : Literal['optim' , 'top'] | Any ,
                 alpha_models : AlphaModel | list[AlphaModel] , 
                 benchmarks : str | None | list = None , 
                 add_lag : int = 0 , 
                 param_groups : dict[Any,dict[str,Any]] = {} ,
                 daily : bool = False ,
                 analytic : bool = True ,
                 attribution : bool = True ,
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
        self.acc_kwargs = {'daily' : daily , 'analytic' : analytic , 'attribution' : attribution}

        self.verbosity = verbosity

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self.alpha_models)} alphas , {len(self.benchmarks)} benchmarks , ' + \
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
            return 'Optimization'
        elif self.category == 'top':
            return 'Generation'
        else:
            return self.category.capitalize()
    
    @property
    def port_name_nchar(self):
        if not hasattr(self , '_port_name_nchar'):
            self._port_name_nchar = np.max([len(builder.full_name()) for builder in self.builders])
        return self._port_name_nchar

    def building(self):
        RISK_MODEL.load_models(self.relevant_dates)
        self.setup_builders()
        self.print_in_optimization('start')
        for self._date in self.relevant_dates:
            for self._builder in self.builders:
                if not self._builder.alpha.has(self._date): continue
                self._builder.build(self._date)
                self.print_in_optimization('loop')
        self.print_in_optimization('end')
        return self
    
    def accounting(self , start : int = -1 , end : int = 99991231):
        t0 = time.time()
        for builder in self.builders:
            with Timer(f'{builder.portfolio.name} accounting') if self.verbosity > 1 else nullcontext():
                builder.accounting(start , end , **self.acc_kwargs)
        if self.verbosity > 0 :
            print(f'Group Accounting Finished , Total time: {time.time()-t0:.2f} secs.')
        return self
    
    def total_account(self):
        assert self.builders , 'No builders to account!'
        df = pd.concat([builder.account for builder in self.builders])
        old_index = list(df.index.names)
        df = df.reset_index().sort_values('model_date')
        df['benchmark'] = pd.Categorical(df['benchmark'] , categories = CONF.CATEGORIES_BENCHMARKS , ordered=True) 

        df = df.set_index(old_index).sort_index()
        torch.save(df , 'account.pt')
        return df
    
    def print_in_optimization(self , where : Literal['start' , 'loop' , 'end']):
        if self.verbosity > 0:
            if where == 'start':
                self.t0 = time.time()
                print(f'{self.category_title} of {str(self)} start!')
                self.opt_count = 0
            elif where == 'loop':
                if self.verbosity > 1 or (self.verbosity > 0 and (self.opt_count % 50 == 0)): 
                    time_cost = {k:np.round(v*1000,2) for k,v in self._builder.creations[-1].time.items()}
                    print(f'{self.category_title} of {self.opt_count:4d}th [{self._builder.portfolio.name:{self.port_name_nchar}s}]' + 
                          f' Finished at {self._date} , time cost (ms) : {time_cost}')
                self.opt_count += 1
            elif where == 'end':
                self.t1 = time.time()
                print(f'Group {self.category_title} Finished , Total time: {self.t1-self.t0:.2f} secs, each optim time: {(self.t1-self.t0)/max(self.opt_count,1):.2f}')
