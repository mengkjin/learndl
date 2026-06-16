"""
RevScreen calculator for Factor Model Portfolio
"""

from __future__ import annotations
import pandas as pd

from src.proj import Base
from src.res.factor.util import StockFactor
from src.res.factor.fmp import PortfolioGroupBuilder
from src.res.factor.util.plot.top_pf import Plotter
from src.res.factor.util.stats import top_pf as Stat

from .test_basics import BaseFactorAnalyticCalculator , BaseFactorAnalyticTest

test_type = Base.TestType.REVSCREEN
plotter = Plotter(test_type.title())

__all__ = [
    'RevScreenCalc' , 
    'FrontFace' , 'Perf_Curve' , 'Perf_Excess' , 'Drawdown' , 'Excess_Drawdown' , 'Perf_Year' ,
    'RevScreenFMPTest'
]

class RevScreenCalc(BaseFactorAnalyticCalculator):
    TEST_TYPE = test_type
    DEFAULT_BENCHMARKS = 'defaults'

    def calc(self , account_df : pd.DataFrame):
        with self.calc_manager(): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account_df)
        return self
    
class FrontFace(RevScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_frontface
    def plotter(self): return plotter.plot_frontface

class Perf_Curve(RevScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_curve
    def plotter(self): return plotter.plot_perf_curve

class Perf_Excess(RevScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess
    def plotter(self): return plotter.plot_perf_excess

class Perf_Year(RevScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_year
    def plotter(self): return plotter.plot_perf_year

class Drawdown(RevScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_drawdown
    def plotter(self): return plotter.plot_perf_drawdown

class Excess_Drawdown(RevScreenCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess_drawdown
    def plotter(self): return plotter.plot_perf_excess_drawdown

class RevScreenFMPTest(BaseFactorAnalyticTest):
    """
    Factor Model PortfolioPerformance Calculator Manager
    Parameters:
        Which tasks to run. Can be any of the following:
        'front_face' : Front Face
        'perf_curve' : Performance Curve
        'perf_excess' : Performance Excess
        'perf_drawdown' : Performance Drawdown
        'perf_year' : Performance Yearly Stats
    """
    TEST_TYPE = test_type
    TASK_LIST : list[type[RevScreenCalc]] = [
        FrontFace , 
        Perf_Curve ,
        Perf_Excess ,
        Drawdown , 
        Perf_Year ,
    ]

    def generate(self , factor: StockFactor , benchmarks : Base.alias.MultipleBenchmark = 'defaults'):
        alpha_models = factor.alpha_models()
        benchmarks = [factor.universe(load = True).get('all').rename('univ')]
        self.update_kwargs()
        self.portfolio_group = PortfolioGroupBuilder(
            test_type , alpha_models , benchmarks , analytic = False , attribution = False , trade_engine = 'yale' , 
            resume = self.resume , resume_path = self.resume_path , caller = self , 
            start = self.start , end = self.end , **self.kwargs)
        self.total_account = self.portfolio_group.build().total_account()

    def calc(self , factor : StockFactor , benchmarks : Base.alias.MultipleBenchmark = 'defaults' , **kwargs):
        self.generate(factor , benchmarks)
        if self.total_account.empty:
            self.logger.error(f'No accounts created for {self.test_name}!')
        else:
            super().calc(self.total_account , **kwargs)
        return self
    
    def update_kwargs(self , **kwargs):
        self.kwargs.update({
            'n_best' : 50 ,
        })
        self.kwargs.update(kwargs)
        return self