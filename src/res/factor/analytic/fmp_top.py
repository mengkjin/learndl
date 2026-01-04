import pandas as pd
from pathlib import Path
from typing import Any , Type
from src.proj import Logger

from src.res.factor.util import Benchmark , StockFactor
from src.res.factor.fmp import PortfolioGroupBuilder
from src.res.factor.util.plot.top_pf import Plotter
from src.res.factor.util.stat import top_pf as Stat

from .test_basics import BaseFactorAnalyticCalculator , BaseFactorAnalyticTest , test_title

test_type = 'top'
plotter = Plotter(test_title(test_type))

class TopCalc(BaseFactorAnalyticCalculator):
    TEST_TYPE = test_type
    DEFAULT_BENCHMARKS = 'defaults'
    def calc(self , account : pd.DataFrame , indent : int = 1 , vb_level : int = 1):
        with self.calc_manager(f'{self.__class__.__name__} calc' , indent = indent , vb_level = vb_level): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        return self
    
class FrontFace(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_frontface
    def plotter(self): return plotter.plot_frontface

class Perf_Curve(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_curve
    def plotter(self): return plotter.plot_perf_curve

class Perf_Excess(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess
    def plotter(self): return plotter.plot_perf_excess

class Drawdown(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_drawdown
    def plotter(self): return plotter.plot_perf_drawdown

class Excess_Drawdown(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess_drawdown
    def plotter(self): return plotter.plot_perf_excess_drawdown

class Perf_Year(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_year
    def plotter(self): return plotter.plot_perf_year

class Perf_Month(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_month
    def plotter(self): return plotter.plot_perf_month

class Exp_Style(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_exp_style
    def plotter(self): return plotter.plot_exp_style

class Exp_Indus(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_exp_indus
    def plotter(self): return plotter.plot_exp_indus

class Attrib_Source(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_attrib_source
    def plotter(self): return plotter.plot_attrib_source

class Attrib_Style(TopCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_attrib_style
    def plotter(self): return plotter.plot_attrib_style

class TopFMPTest(BaseFactorAnalyticTest):
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
            'exp_style' : Style Exposure
            'exp_indus' : Industry Deviation
            'attrib_source' : Attribution Source
            'attrib_style' : Attribution Style
    '''
    TEST_TYPE = test_type
    TASK_LIST : list[Type[TopCalc]] = [
        FrontFace , 
        Perf_Curve ,
        Perf_Excess ,
        Drawdown , 
        Excess_Drawdown ,
        Perf_Year ,
        Exp_Style ,
        Exp_Indus ,
        Attrib_Source ,
        Attrib_Style ,
    ]

    def generate(self , factor: StockFactor , benchmarks: list[Benchmark|Any] | Any = 'defaults' , 
                 n_bests = [20,30,50,100] , indent : int = 0 , vb_level : int = 1):
        alpha_models = factor.alpha_models()
        benchmarks = Benchmark.get_benchmarks(benchmarks)
        self.update_kwargs(n_bests = n_bests)
        self.portfolio_group = PortfolioGroupBuilder(
            'top' , alpha_models , benchmarks , resume = self.resume , resume_path = self.portfolio_resume_path , 
            caller = self , start_dt = self.start_dt , end_dt = self.end_dt , indent = indent , vb_level = vb_level , **self.kwargs)
        self.account = self.portfolio_group.build().accounts()

    def calc(self , factor : StockFactor , benchmark : list[Benchmark|Any] | Any | None = 'defaults' ,
             n_bests = [20,30,50,100] , indent : int = 0 , vb_level : int = 1 , **kwargs):
        self.generate(factor , benchmark , n_bests = n_bests , indent = indent , vb_level = vb_level)
        with Logger.Timer(f'{self.__class__.__name__}.calc' , indent = indent , vb_level = vb_level , enter_vb_level = vb_level + 1):
            for task in self.tasks.values():  
                task.calc(self.account , indent = indent + 1 , vb_level = vb_level + 1) 
        return self
    
    def update_kwargs(self , n_bests = [20,30,50,100] , **kwargs):
        self.kwargs.update({
            'add_lag': 0 ,# 1 if any([task in self.tasks for task in ['perf_lag']]) else 0 ,
            'analytic':any([task in self.tasks for task in ['exp_style' , 'exp_indus']]) ,
            'attribution':any([task in self.tasks for task in ['attrib_source' , 'attrib_style']]) ,
        })
        self.kwargs['param_groups'] = {f'Top{n:3d}':{'n_best':n} for n in n_bests}
        self.kwargs.update(kwargs)
        return self

    def save(self , path : Path | str):
        """save intermediate data to path for future use"""
        ...
        
    def load(self , path : Path | str):
        """load intermediate data from path for future use"""
        ...