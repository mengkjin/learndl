import itertools , time
import pandas as pd
import numpy as np

from contextlib import nullcontext
from dataclasses import dataclass , field
from typing import Any , Literal

from src.basic import Timer
from src.basic.conf import ROUNDING_RETURN , ROUNDING_TURNOVER , TRADE_COST
from src.data import DATAVENDOR
from src.factor.util import Portfolio , Benchmark , AlphaModel , RISK_MODEL , PortfolioOptimizer , PortOptimResult , Port
# from ...util import Portfolio , Benchmark , AlphaModel , RISK_MODEL , PortfolioOptimizer , PortOptimResult , Port

@dataclass
class PortfolioBuilder:
    category  : Literal['optim' , 'top'] | Any
    name : str
    alpha : AlphaModel
    benchmark : Portfolio | Any = None
    lag       : int = 0
    portfolio : Portfolio | Any = None
    optimizer : PortfolioOptimizer | Any = None
    optimrslt : list[PortOptimResult] = field(default_factory=list)
    account   : pd.DataFrame | Any = None

    def __post_init__(self):
        if self.portfolio is None: self.portfolio = Portfolio(self.name)
        if self.benchmark is None: self.benchmark = Portfolio(is_default=True)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha=\'{self.alpha.name}\',category=\'{self.category}\','+\
            f'benchmark=\'{self.benchmark.name}\',lag={self.lag},'+\
            f'{len(self.portfolio)} fmp\'s,'+'not '* (self.account is None) + 'accounted)'
    
    def setup(self , prob_type : Literal['linprog', 'quadprog', 'socp'] | Any = 'linprog' , config_path : str | None = None):
        if self.category == 'optim':
            self.optimizer = PortfolioOptimizer(prob_type).setup_optimizer(self.name , config_path)
        elif self.category == 'top':
            pass
        return self
        
    def build(self , date : int):
        if self.category == 'optim':
            self.build_optim(date)
        else:
            self.build_top(date)
        return self
    
    def build_optim(self , date : int):
        '''Optimize at a single date'''
        assert self.alpha.has(date) , f'{self.alpha.name} has no data at {date}'
        init_port = self.portfolio.last_port(date)
        opt = self.optimizer.optimize(date , self.alpha.get(date , lag = self.lag) , self.benchmark , init_port)
        self.optimrslt.append(opt)
        self.portfolio.append(opt.port)
        return self
    
    def build_top(self , date : int):
        assert self.alpha.has(date) , f'{self.alpha.name} has no data at {date}'
        init_port = self.portfolio.last_port(date)

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

        self.account = pd.DataFrame({'model_date':np.concatenate([[-1],model_dates]) , 
                                     'st':np.concatenate([[model_dates[0]],period_st]) , 
                                     'ed':np.concatenate([[model_dates[0]],period_ed]) ,
                                     'pf':0. , 'bm':0. , 'turn':0. , 'excess':0. ,
                                     'analytic':None , 'attribution':None}).set_index('model_date')

        port_old = Port.none_port(model_dates[0])
        for date , ed in zip(model_dates , period_ed):
            port_new = self.portfolio.get(date) if self.portfolio.has(date) else port_old
            bench = Port.none_port(date) if self.benchmark is None else self.benchmark.get(date , True)

            turn = np.round(port_new.turnover(port_old),ROUNDING_TURNOVER)
            self.account.loc[date , ['pf' , 'bm' , 'turn']] = \
                [np.round(port_new.fut_ret(ed) , ROUNDING_RETURN) , np.round(bench.fut_ret(ed),ROUNDING_RETURN) , turn]
            
            if analytic and self.lag == 0: 
                self.account.loc[date , 'analytic']    = RISK_MODEL.get(date).analyze(port_new , bench , port_old) #type:ignore
            if attribution and self.lag == 0: 
                self.account.loc[date , 'attribution'] = RISK_MODEL.get(date).attribute(port_new , bench , ed , turn * TRADE_COST)  #type:ignore
            port_old = port_new.evolve_to_date(ed , inplace=False)

        self.account['pf']  = self.account['pf'] - self.account['turn'] * TRADE_COST
        self.account['excess'] = self.account['pf'] - self.account['bm']
        return self

class PortfolioBuilderGroup:
    def __init__(self , category : Literal['optim' , 'top'] | Any ,
                 alpha_models : AlphaModel | list[AlphaModel] , 
                 benchmarks : str | None | list = None , 
                 add_lag : int = 1 , 
                 optim_config_path : str | None = None , 
                 optim_prob_type : Literal['linprog', 'quadprog', 'socp'] | Any = 'linprog', 
                 acc_daily : bool = False , acc_analytic : bool = True , acc_attribution : bool = True ,
                 verbosity : int = 1):
        self.builders : list[PortfolioBuilder] = []
        self.category = category

        assert alpha_models , f'alpha_models must has elements!'
        self.alpha_models = alpha_models if isinstance(alpha_models , list) else [alpha_models]
        self.relevant_dates = np.unique(np.concatenate([amodel.available_dates() for amodel in self.alpha_models]))

        self.benchmarks = Benchmark.get_benchmarks(benchmarks)

        assert add_lag > 0 , add_lag
        self.lags = [0 , add_lag]

        self.optim_config_path = optim_config_path
        self.optim_prob_type = optim_prob_type

        self.acc_daily = acc_daily
        self.acc_analytic = acc_analytic
        self.acc_attribution = acc_attribution

        self.verbosity = verbosity

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self.alpha_models)} alphas , {len(self.benchmarks)} benchmarks , ' + \
               f'{len(self.lags)} lags , {len(self.relevant_dates)} dates , ' + \
               f'({len(self.alpha_models) * len(self.benchmarks) * len(self.lags) * len(self.relevant_dates)} builds)'
    
    def setup_builders(self):
        self.builders.clear()
        for (alpha , lag , bench) in itertools.product(self.alpha_models , self.lags , self.benchmarks):
            port_name = f'{alpha.name}.{bench.name}.lag{lag}'
            builder = PortfolioBuilder(self.category , port_name , alpha , benchmark = bench , lag = lag).\
                setup(self.optim_prob_type , self.optim_config_path)
            self.builders.append(builder)
        return self
    
    @property
    def port_name_nchar(self):
        if not hasattr(self , '_port_name_nchar'):
            self._port_name_nchar = np.max([len(pt.name) for pt in self.builders])
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
        for builder in self.builders:
            with Timer(f'{builder.portfolio.name} accounting') if self.verbosity > 0 else nullcontext():
                builder.accounting(start , end , self.acc_daily , self.acc_analytic , self.acc_attribution)
        return self
    
    def total_account(self):
        assert self.builders , 'No builders to account!'
        df = pd.concat([builder.account.assign(factor_name = builder.alpha.name , benchmark = builder.benchmark.name , lag = builder.lag) 
                        for builder in self.builders]).reset_index().rename(columns={'st':'start' , 'ed':'end'})
        return df
    
    def print_in_optimization(self , where : Literal['start' , 'loop' , 'end']):
        if self.verbosity > 0:
            if where == 'start':
                self.t0 = time.time()
                print(f'Optimization of {str(self)} start!')
                self.opt_count = 0
            elif where == 'loop':
                if self.verbosity > 1 or (self.verbosity > 0 and (self.opt_count % 50 == 0)): 
                    time_cost = {k:np.round(v*1000,2) for k,v in self._builder.optimrslt[-1].time.items()}
                    print(f'Done Optimize {self.opt_count:4d}th [{self._builder.portfolio.name:{self.port_name_nchar}s}]' + 
                          f' at {self._date} , time cost (ms) : {time_cost}')
                self.opt_count += 1
            elif where == 'end':
                self.t1 = time.time()
                print(f'Group optimization Finished , Total time: {self.t1-self.t0:.2f} secs, each optim time: {(self.t1-self.t0)/max(self.opt_count,1):.2f}')
