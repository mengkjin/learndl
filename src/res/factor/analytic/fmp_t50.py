import pandas as pd
from typing import Any , Type

from src.proj import Logger

from src.res.factor.util import StockFactor
from src.res.factor.fmp import PortfolioGroupBuilder
from src.res.factor.util.plot.top_pf import Plotter
from src.res.factor.util.stat import top_pf as Stat

from .test_basics import BaseFactorAnalyticCalculator , BaseFactorAnalyticTest , test_title

test_type = 't50'
plotter = Plotter(test_title(test_type))

class T50Calc(BaseFactorAnalyticCalculator):
    TEST_TYPE = 't50'
    DEFAULT_BENCHMARKS = 'defaults'

    def calc(self , account_df : pd.DataFrame , indent : int = 1 , vb_level : int = 1):
        with self.calc_manager(f'{self.__class__.__name__} calc' , indent = indent , vb_level = vb_level): 
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
    TASK_LIST : list[Type[T50Calc]] = [
        FrontFace , 
        Perf_Curve ,
        Perf_Excess ,
        Drawdown , 
        Perf_Year ,
    ]

    def generate(self , factor: StockFactor , benchmark : Any = 'defaults' , indent : int = 0 , vb_level : int = 1):
        alpha_models = factor.alpha_models()
        benchmarks = [factor.universe(load = True).to_portfolio().rename('univ')]
        self.update_kwargs()
        self.portfolio_group = PortfolioGroupBuilder(
            'top' , alpha_models , benchmarks , analytic = False , attribution = False , trade_engine = 'yale' , 
            resume = self.resume , resume_path = self.resume_path , caller = self ,
            start_dt = self.start_dt , end_dt = self.end_dt , indent = indent , vb_level = vb_level , **self.kwargs)
        self.total_account = self.portfolio_group.build().total_account()

    def calc(self , factor : StockFactor , benchmark : Any = 'defaults' , indent : int = 0 , vb_level : int = 1 , **kwargs):
        self.generate(factor , benchmark , indent = indent , vb_level = vb_level)
        if self.total_account.empty:
            Logger.error(f'No accounts created for {self.test_name}!')
        else:
            super().calc(self.total_account , indent = indent , vb_level = vb_level , **kwargs)
        return self
    
    def update_kwargs(self , **kwargs):
        self.kwargs.update({
            'n_best' : 50 , 
        })
        self.kwargs.update(kwargs)
        return self