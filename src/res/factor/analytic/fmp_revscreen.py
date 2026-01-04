import pandas as pd
from typing import Any , Type

from src.proj import Logger

from src.res.factor.util import StockFactor
from src.res.factor.fmp import PortfolioGroupBuilder
from src.res.factor.util.plot.top_pf import Plotter
from src.res.factor.util.stat import top_pf as Stat

from .test_basics import BaseFactorAnalyticCalculator , BaseFactorAnalyticTest , test_title

test_type = 'revscreen'
plotter = Plotter(test_title(test_type))

class RevScreenCalc(BaseFactorAnalyticCalculator):
    TEST_TYPE = test_type
    DEFAULT_BENCHMARKS = 'defaults'

    def calc(self , account : pd.DataFrame , indent : int = 0 , vb_level : int = 1):
        with self.calc_manager(f'{self.__class__.__name__} calc' , indent = indent , vb_level = vb_level): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
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
    TASK_LIST : list[Type[RevScreenCalc]] = [
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
            'revscreen' , alpha_models , benchmarks , analytic = False , attribution = False , trade_engine = 'yale' , 
            resume = self.resume , resume_path = self.portfolio_resume_path , caller = self , 
            start_dt = self.start_dt , end_dt = self.end_dt , indent = indent , vb_level = vb_level , **self.kwargs)
        self.account = self.portfolio_group.build().accounts()

    def calc(self , factor : StockFactor , benchmark : Any = 'defaults' , indent : int = 0 , vb_level : int = 1 , **kwargs):
        self.generate(factor , benchmark , indent = indent , vb_level = vb_level)
        with Logger.Timer(f'{self.__class__.__name__}.calc' , indent = indent , vb_level = vb_level , enter_vb_level = vb_level + 1):
            for task in self.tasks.values():  
                task.calc(self.account , indent = indent + 1 , vb_level = vb_level + 1) 
        return self
    
    def update_kwargs(self , **kwargs):
        self.kwargs.update({
            'n_best' : 50 ,
        })
        self.kwargs.update(kwargs)
        return self