import pandas as pd
import numpy as np
from typing import Any , Type

from src.proj import Timer

from src.res.factor.util import StockFactor , Universe
from src.res.factor.fmp import PortfolioGroupBuilder
from src.res.factor.util.plot.top_pf import Plotter
from src.res.factor.util.stat import top_pf as Stat

from .test_basics import BaseFactorAnalyticCalculator , BaseFactorAnalyticTest , test_title

test_type = 'screen'
plotter = Plotter(test_title(test_type))

class ScreenCalc(BaseFactorAnalyticCalculator):
    TEST_TYPE = test_type
    DEFAULT_BENCHMARKS = 'defaults'

    def calc(self , account : pd.DataFrame , verbosity = 0):
        with self.calc_manager(f'  --> {self.__class__.__name__} calc' , verbosity = verbosity): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        return self
    
class FrontFace(ScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_frontface
    def plotter(self): return plotter.plot_frontface

class Perf_Curve(ScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_curve
    def plotter(self): return plotter.plot_perf_curve

class Perf_Excess(ScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess
    def plotter(self): return plotter.plot_perf_excess

class Drawdown(ScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_drawdown
    def plotter(self): return plotter.plot_perf_drawdown

class Excess_Drawdown(ScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess_drawdown
    def plotter(self): return plotter.plot_perf_excess_drawdown

class Perf_Year(ScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_year
    def plotter(self): return plotter.plot_perf_year


class ScreenFMPTest(BaseFactorAnalyticTest):
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
    TEST_TYPE = test_type
    TASK_LIST : list[Type[ScreenCalc]] = [
        FrontFace , 
        Perf_Curve ,
        Perf_Excess ,
        Drawdown , 
        Perf_Year ,
    ]

    def generate(self , factor: StockFactor , benchmark : Any = 'defaults' , verbosity = 2 , **kwargs):
        alpha_models = factor.alpha_models()
        dates = np.unique(np.concatenate([alpha.available_dates() for alpha in alpha_models]))
        universe = Universe('top-1000')
        benchmarks = [universe.to_portfolio(dates).rename('univ')]
        self.update_kwargs(verbosity = verbosity)
        self.portfolio_group = PortfolioGroupBuilder('screen' , alpha_models , benchmarks , analytic = False , attribution = False , trade_engine = 'yale' , resume = self.resume , resume_path = self.resume_path , caller = self , **self.kwargs)
        self.account = self.portfolio_group.build().accounts()

    def calc(self , factor : StockFactor , benchmark : Any = 'defaults' , verbosity = 1 , **kwargs):
        self.generate(factor , benchmark , verbosity = verbosity)
        with Timer(f'{self.__class__.__name__}.calc' , silent = verbosity < 1):
            for task in self.tasks.values():  
                task.calc(self.account , verbosity = verbosity - 1) 
        return self
    
    def update_kwargs(self , **kwargs):
        self.kwargs.update({
            'n_best' : 50 ,
        })
        self.kwargs.update(kwargs)
        return self