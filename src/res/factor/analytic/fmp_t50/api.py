import numpy as np
from typing import Any , Type

from src.proj import Timer
from . import calculator as Calc
from ..test_manager import BaseTestManager
from ...util import StockFactor , Universe
from ...fmp import PortfolioBuilderGroup

__all__ = ['FmpT50Manager']

class FmpT50Manager(BaseTestManager):
    '''
    Factor Model PortfolioPerformance Calculator Manager
    Parameters:
        which : str | list[str] | Literal['all']
            Which tasks to run. Can be any of the following:
            'front_face' : Front Face
            'perf_curve' : Performance Curve
            'perf_excess' : Performance Excess
            'perf_drawdown' : Performance Drawdown
            'perf_year' : Performance Yearly Stats
    '''
    TASK_TYPE = 't50'
    TASK_LIST : list[Type[Calc.BaseT50PortCalc]] = [
        Calc.FrontFace , 
        Calc.Perf_Curve ,
        Calc.Perf_Excess ,
        Calc.Drawdown , 
        Calc.Perf_Year ,
    ]

    def generate(self , factor: StockFactor , benchmark : Any = 'defaults' , verbosity = 2 , **kwargs):
        alpha_models = factor.alpha_models()
        dates = np.unique(np.concatenate([alpha.available_dates() for alpha in alpha_models]))
        universe = Universe('top-1000')
        benchmarks = [universe.to_portfolio(dates).rename('univ')]
        self.update_kwargs(verbosity = verbosity)
        self.portfolio_group = PortfolioBuilderGroup('top' , alpha_models , benchmarks , analytic = False , attribution = False , trade_engine = 'yale' , **self.kwargs)
        self.account = self.portfolio_group.building().accounting().total_account()

    def calc(self , factor : StockFactor , benchmark : Any = 'defaults' , verbosity = 1 , **kwargs):
        with Timer(f'{self.__class__.__name__} calc' , silent = verbosity < 1):
            self.generate(factor , benchmark , verbosity = verbosity)
            for task in self.tasks.values():  
                task.calc(self.account , verbosity = verbosity - 1) 
        return self
    
    def update_kwargs(self , **kwargs):
        self.kwargs.update({
            'n_best' : 50 ,
        })
        self.kwargs.update(kwargs)
        return self