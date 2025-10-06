import pandas as pd

from . import stat as Stat
from . import plot as Plot
from ..test_manager import BaseCalculator

class BaseScreenPortCalc(BaseCalculator):
    TASK_TYPE = 'screen'
    DEFAULT_BENCHMARKS = 'defaults'
    DEFAULT_TITLE_GROUP = 'Screen Port'

    def calc(self , account : pd.DataFrame , verbosity = 0):
        with self.suppress_warnings(): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        if verbosity > 0: 
            print(f'    --->{self.__class__.__name__} calc Finished!')
        return self
    
class Screen_FrontFace(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_top_frontface
    def plotter(self): return Plot.plot_top_frontface

class Screen_Perf_Curve(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_top_perf_curve
    def plotter(self): return Plot.plot_top_perf_curve

class Screen_Perf_Excess(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_top_perf_excess
    def plotter(self): return Plot.plot_top_perf_excess

class Screen_Perf_Drawdown(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_top_perf_drawdown
    def plotter(self): return Plot.plot_top_perf_drawdown

class T50_Perf_Excess_Drawdown(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_top_perf_excess_drawdown
    def plotter(self): return Plot.plot_top_perf_excess_drawdown

class Screen_Perf_Year(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_top_perf_year
    def plotter(self): return Plot.plot_top_perf_year
