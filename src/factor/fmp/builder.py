import itertools , time
import pandas as pd
import numpy as np

from dataclasses import dataclass , field
from typing import Any , Literal

from ..optimizer.api import PortfolioOptimizer , PortOptimResult
from ..util import AlphaModel , RISK_MODEL , Portfolio , Benchmark

@dataclass
class PortOptimTuple:
    name : str
    alpha : AlphaModel
    portfolio : Portfolio
    benchmark : Portfolio
    optimizer : PortfolioOptimizer
    lag       : int = 0
    optimrslt : list[PortOptimResult] = field(default_factory=list)
    account   : pd.DataFrame | Any = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha=\'{self.alpha.name}\',benchmark=\'{self.benchmark.name}\',lag={self.lag},'+\
            f'{len(self.portfolio)} fmp\'s,'+'not '* (self.account is None) + 'accounted)'

def group_optimize(alpha_models : AlphaModel | list[AlphaModel] , benchmarks : str | None | list = None , 
                   add_lag : int = 1 , config_path : str | None = None , 
                   prob_type : Literal['linprog', 'quadprog', 'socp'] = 'linprog', verbosity : int = 1):
    '''
    Create Factor Model Portfolios for AlphaModels , using given or default benchmarks
    '''
    assert alpha_models , f'alpha_models must has elements!'
    t0 = time.time()
    if not isinstance(alpha_models , list): alpha_models = [alpha_models]
    assert add_lag > 0 , add_lag
    lags = [0 , add_lag]

    benches = Benchmark.get_benchmarks(benchmarks)
    relevant_dates = np.unique(np.concatenate([amodel.available_dates() for amodel in alpha_models]))
    if verbosity > 0: 
        print(f'Group optimization of {len(alpha_models)} alphas , {len(benches)} benchmarks , ' + 
              f'{len(lags)} lags , {len(relevant_dates)} dates , ' +
              f'({len(alpha_models) * len(benches) * len(lags) * len(relevant_dates)} opts) start!')

    RISK_MODEL.load_models(relevant_dates)

    port_iter = list(itertools.product(alpha_models , lags , benches))
    port_tuples : list[PortOptimTuple] = []

    for (alpha , lag , bench) in port_iter:
        port_name = f'{alpha.name}.{bench.name}' + f'.{lag}' * (lag > 0) 
        port_tuple = PortOptimTuple(port_name , alpha , Portfolio(port_name) , 
                                    Portfolio(is_default=True) if bench is None else bench , 
                                    PortfolioOptimizer(prob_type).setup_optimizer(port_name , config_path) , lag = lag)
        port_tuples.append(port_tuple)

    port_name_len = np.max([len(pt.name) for pt in port_tuples])
    t1 = time.time()
    opt_count = 0
    for date in relevant_dates:
        for pt in port_tuples:
            if not alpha.has(date): continue
            init_port = pt.optimrslt[-1].port.evolve_to_date(date) if pt.optimrslt else None
            opt = pt.optimizer.optimize(date , pt.alpha.get(date , lag = pt.lag) , pt.benchmark , init_port)
            pt.optimrslt.append(opt)
            pt.portfolio.append(opt.port)
            if verbosity > 1 or (verbosity > 0 and (opt_count % 50 == 0)): 
                time_cost = {k:np.round(v*1000,2) for k,v in opt.time.items()}
                print(f'Done Optimize {opt_count:4d}th [{pt.portfolio.name:{port_name_len}s}] at {date} , time cost (ms) : {time_cost}')
            opt_count += 1
    
    t2 = time.time()
    if verbosity > 0: 
        print(f'Group optimization Finished , Total time: {t2-t0:.2f} secs, Setup time: {t1-t0:.2f} secs, ' + 
            f'Calc time: {t2-t1:.2f} secs, Each optim time: {(t2-t1)/max(opt_count,1):.2f}')
    return port_tuples