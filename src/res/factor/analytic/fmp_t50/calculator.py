import pandas as pd

from . import stat as Stat
from . import plot as Plot
from ..test_manager import BaseCalculator

class BaseT50PortCalc(BaseCalculator):
    TASK_TYPE = 't50'
    DEFAULT_BENCHMARKS = 'defaults'
    def calc(self , account : pd.DataFrame , verbosity = 0):
        with self.suppress_warnings(): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        if verbosity > 0: 
            print(f'    --->{self.__class__.__name__} calc Finished!')
        return self
    
class T50_FrontFace(BaseT50PortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_frontface
    def plotter(self): return Plot.plot_top_frontface

class T50_Perf_Curve(BaseT50PortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_curve
    def plotter(self): return Plot.plot_top_perf_curve

class T50_Perf_Excess(BaseT50PortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_excess
    def plotter(self): return Plot.plot_top_perf_excess

class T50_Perf_Drawdown(BaseT50PortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_drawdown
    def plotter(self): return Plot.plot_top_perf_drawdown

class T50_Perf_Excess_Drawdown(BaseT50PortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_excess_drawdown
    def plotter(self): return Plot.plot_top_perf_excess_drawdown

class T50_Perf_Year(BaseT50PortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_year
    def plotter(self): return Plot.plot_top_perf_year
