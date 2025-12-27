import pandas as pd

from typing import Any , Literal , Type

from src.proj import Logger

from src.res.factor.util import Benchmark , StockFactor
from src.res.factor.fmp import PortfolioGroupBuilder
from src.res.factor.util.plot.optim_pf import Plotter
from src.res.factor.util.stat import optim_pf as Stat

from .test_basics import BaseFactorAnalyticCalculator , BaseFactorAnalyticTest , test_title

test_type = 'optim'
plotter = Plotter(test_title(test_type))

class OptimCalc(BaseFactorAnalyticCalculator):
    TEST_TYPE = test_type
    DEFAULT_BENCHMARKS = 'defaults'

    def calc(self , account : pd.DataFrame , indent : int = 0 , vb_level : int = 1):
        with self.calc_manager(f'{self.__class__.__name__} calc' , indent = indent , vb_level = vb_level): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        return self
    
class FrontFace(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_frontface
    def plotter(self): return plotter.plot_frontface

class Perf_Curve(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_curve
    def plotter(self): return plotter.plot_perf_curve

class Drawdown(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_drawdown
    def plotter(self): return plotter.plot_perf_drawdown

class Excess_Drawdown(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess_drawdown
    def plotter(self): return plotter.plot_perf_excess_drawdown

class Perf_Lag(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_lag
    def plotter(self): return plotter.plot_perf_lag

class Perf_Year(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_year
    def plotter(self): return plotter.plot_perf_year

class Perf_Month(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_month
    def plotter(self): return plotter.plot_perf_month
class Exp_Style(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_exp_style
    def plotter(self): return plotter.plot_exp_style

class Exp_Indus(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_exp_indus
    def plotter(self): return plotter.plot_exp_indus

class Attrib_Source(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_attrib_source
    def plotter(self): return plotter.plot_attrib_source

class Attrib_Style(OptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_attrib_style
    def plotter(self): return plotter.plot_attrib_style

class OptimFMPTest(BaseFactorAnalyticTest):
    '''
    Factor Model PortfolioPerformance Calculator Manager
    Parameters:
        which : str | list[str] | Literal['all']
            Which tasks to run. Can be any of the following:
            'prefix' : Prefix
            'perf_curve' : Performance Curve
            'perf_drawdown' : Performance Drawdown
            'perf_year' : Performance Yearly Stats
            'perf_month' : Performance Monthly Stats
            'perf_lag' : Performance Lag Curve
            'exp_style' : Style Exposure
            'exp_indus' : Industry Deviation
            'attrib_source' : Attribution Source
            'attrib_style' : Attribution Style
    '''
    TEST_TYPE = 'optim'
    TASK_LIST : list[Type[OptimCalc]] = [
        FrontFace , 
        Perf_Curve ,
        Drawdown , 
        Excess_Drawdown , 
        Perf_Year ,
        Perf_Month ,
        Perf_Lag ,
        Exp_Style ,
        Exp_Indus ,
        Attrib_Source ,
        Attrib_Style ,
    ]

    def optim(self , factor: StockFactor , benchmarks: list[Benchmark|Any] | Any = 'defaults' , 
              add_lag = 1 , optim_config = None , indent : int = 0 , vb_level : int = 1):
        alpha_models = factor.alpha_models()
        benchmarks = Benchmark.get_benchmarks(benchmarks)
        self.update_kwargs(add_lag = add_lag , optim_config = optim_config)
        self.portfolio_group = PortfolioGroupBuilder(
            'optim' , alpha_models , benchmarks , resume = self.resume , resume_path = self.resume_path , caller = self , 
            start_dt = self.start_dt , end_dt = self.end_dt , indent = indent , vb_level = vb_level , **self.kwargs)
        self.account = self.portfolio_group.build().accounts()

    def calc(self , factor : StockFactor , benchmark : list[Benchmark|Any] | Any | None = 'defaults' ,
             add_lag = 1 , optim_config : str | Literal['default' , 'custome'] | None = None , 
             indent : int = 0 , vb_level : int = 1 , **kwargs):
        self.optim(factor , benchmark , add_lag = add_lag ,optim_config = optim_config , indent = indent , vb_level = vb_level)
        with Logger.Timer(f'{self.__class__.__name__}.calc' , indent = indent , vb_level = vb_level , enter_vb_level = vb_level + 1):
            for task in self.tasks.values():  
                task.calc(self.account , indent = indent + 1 , vb_level = vb_level + 1) 
        return self
    
    def update_kwargs(self , add_lag = 1 , **kwargs):
        self.kwargs.update({
            'add_lag': 1 if any([task in self.tasks for task in ['perf_lag']]) else add_lag ,
            'analytic':any([task in self.tasks for task in ['exp_style' , 'exp_indus']]) ,
            'attribution':any([task in self.tasks for task in ['attrib_source' , 'attrib_style']])})
        self.kwargs.update(kwargs)
        return self
