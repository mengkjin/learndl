"""
T50 calculator for Factor Model Portfolio
"""
from __future__ import annotations
import pandas as pd

from src.proj import Base
from src.res.factor.util import StockFactor
from src.res.factor.fmp import PortfolioGroupBuilder
from src.res.factor.util.plot.top_pf import Plotter
from src.res.factor.util.stats import top_pf as Stat

from .test_basics import BaseFactorAnalyticCalculator , BaseFactorAnalyticTest

test_type = Base.TestType.T50
plotter = Plotter(test_type.title())

__all__ = [
    'T50Calc' , 
    'FrontFace' , 'Perf_Curve' , 'Perf_Excess' , 'Drawdown' , 'Excess_Drawdown' , 'Perf_Year' ,
    'T50FMPTest'
]

class T50Calc(BaseFactorAnalyticCalculator):
    TEST_TYPE = test_type
    DEFAULT_BENCHMARKS = 'defaults'

    def calc(self , account_df : pd.DataFrame):
        with self.calc_manager(): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account_df)
        return self
    
class FrontFace(T50Calc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_frontface
    def plotter(self): return plotter.plot_frontface

class Perf_Curve(T50Calc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_curve
    def plotter(self): return plotter.plot_perf_curve

class Perf_Excess(T50Calc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess
    def plotter(self): return plotter.plot_perf_excess

class Drawdown(T50Calc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_drawdown
    def plotter(self): return plotter.plot_perf_drawdown

class Excess_Drawdown(T50Calc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess_drawdown
    def plotter(self): return plotter.plot_perf_excess_drawdown

class Perf_Year(T50Calc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_year
    def plotter(self): return plotter.plot_perf_year

class T50FMPTest(BaseFactorAnalyticTest):
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
    TASK_LIST : list[type[T50Calc]] = [
        FrontFace , 
        Perf_Curve ,
        Perf_Excess ,
        Drawdown , 
        Perf_Year ,
    ]

    def generate(self , factor: StockFactor , benchmarks : Base.alias.MultipleBenchmark = 'defaults'):
        alpha_models = factor.alpha_models()
        universe_portfolio = self.kwargs.pop('universe_portfolio', None)
        if universe_portfolio is None:
            from src.res.factor.util.classes.universe import Universe
            universe_portfolio = Universe('top-1000').get(factor.date).rename('univ')
        benchmarks = [universe_portfolio]
        self.update_kwargs()
        from src.res.factor.fmp.generator_array import run_generator
        builder_kwargs = self.kwargs.copy()
        backend = builder_kwargs.pop('t50_backend', 'auto')
        generic_backend = builder_kwargs.pop('generator_backend', None)
        prepared_root = builder_kwargs.pop('prepared_market_root', None)
        if generic_backend is not None and 't50_backend' in self.kwargs and generic_backend != backend:
            raise ValueError('conflicting generator_backend and t50_backend')
        self.portfolio_group = PortfolioGroupBuilder(
            'top' , alpha_models , benchmarks , analytic = False , attribution = False , trade_engine = 'yale' , 
            resume = self.resume , resume_path = self.resume_path , caller = self ,
            start = self.start , end = self.end , **builder_kwargs)
        self.total_account = run_generator(
            self.portfolio_group, generic_backend or backend, prepared_root=prepared_root)


    def calc(self , factor : StockFactor , benchmarks : Base.alias.MultipleBenchmark = 'defaults' , **kwargs):
        self.generate(factor , benchmarks)
        if self.total_account.empty:
            self.logger.error(f'No accounts created for {self.test_name}!')
        else:
            super().calc(self.total_account , **kwargs)
        return self
    
    def update_kwargs(self , **kwargs):
        self.kwargs.setdefault('n_best', 50)
        self.kwargs.update(kwargs)
        return self
