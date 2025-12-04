import pandas as pd

from ..test_manager import BaseCalculator
from src.res.factor.util.plot.optim_pf import Plotter
from src.res.factor.util.stat import optim_pf as Stat

plotter = Plotter('Optim Port')

class BaseOptimCalc(BaseCalculator):
    TASK_TYPE = 'optim'
    DEFAULT_BENCHMARKS = 'defaults'
    DEFAULT_TITLE_GROUP = 'Optim Port'

    def calc(self , account : pd.DataFrame , verbosity = 0):
        with self.calc_manager(f'    --->{self.__class__.__name__} calc' , verbosity = verbosity): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        return self
    
class Optim_FrontFace(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_frontface
    def plotter(self): return plotter.plot_frontface

class Optim_Perf_Curve(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_curve
    def plotter(self): return plotter.plot_perf_curve

class Optim_Perf_Drawdown(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_drawdown
    def plotter(self): return plotter.plot_perf_drawdown

class Optim_Perf_Excess_Drawdown(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess_drawdown
    def plotter(self): return plotter.plot_perf_excess_drawdown

class Optim_Perf_Lag(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_lag
    def plotter(self): return plotter.plot_perf_lag

class Optim_Perf_Year(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_year
    def plotter(self): return plotter.plot_perf_year

class Optim_Perf_Month(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_month
    def plotter(self): return plotter.plot_perf_month
class Optim_Exp_Style(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_exp_style
    def plotter(self): return plotter.plot_exp_style

class Optim_Exp_Indus(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_exp_indus
    def plotter(self): return plotter.plot_exp_indus

class Optim_Attrib_Source(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_attrib_source
    def plotter(self): return plotter.plot_attrib_source

class Optim_Attrib_Style(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_attrib_style
    def plotter(self): return plotter.plot_attrib_style