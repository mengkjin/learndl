import pandas as pd

from . import stat as Stat
from . import plot as Plot
from ..test_manager import BaseCalculator

class BaseOptimCalc(BaseCalculator):
    TASK_TYPE = 'optim'
    DEFAULT_BENCHMARKS = 'defaults'
    def calc(self , account : pd.DataFrame , verbosity = 0):
        with self.suppress_warnings(): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        if verbosity > 0: print(f'    --->{self.__class__.__name__} calc Finished!')
        return self
    
class Optim_FrontFace(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_frontface
    def plotter(self): return Plot.plot_optim_frontface

class Optim_Perf_Curve(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_perf_curve
    def plotter(self): return Plot.plot_optim_perf_curve

class Optim_Perf_Drawdown(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_perf_drawdown
    def plotter(self): return Plot.plot_optim_perf_drawdown

class Optim_Perf_Lag(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_perf_lag
    def plotter(self): return Plot.plot_optim_perf_lag

class Optim_Perf_Year(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_perf_year
    def plotter(self): return Plot.plot_optim_perf_year

class Optim_Perf_Month(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_perf_month
    def plotter(self): return Plot.plot_optim_perf_month
class Optim_Exp_Style(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_exp_style
    def plotter(self): return Plot.plot_optim_exp_style

class Optim_Exp_Indus(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_exp_indus
    def plotter(self): return Plot.plot_optim_exp_indus

class Optim_Attrib_Source(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_attrib_source
    def plotter(self): return Plot.plot_optim_attrib_source

class Optim_Attrib_Style(BaseOptimCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_optim_attrib_style
    def plotter(self): return Plot.plot_optim_attrib_style