import pandas as pd

from ..test_manager import BaseCalculator
from src.res.factor.util.plot.top_pf import Plotter
from src.res.factor.util.stat import top_pf as Stat

plotter = Plotter('Screen Port')

class BaseScreenPortCalc(BaseCalculator):
    TASK_TYPE = 'screen'
    DEFAULT_BENCHMARKS = 'defaults'
    DEFAULT_TITLE_GROUP = 'Screen Port'

    def calc(self , account : pd.DataFrame , verbosity = 0):
        with self.calc_manager(f'    --->{self.__class__.__name__} calc' , verbosity = verbosity): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        return self
    
class FrontFace(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_frontface
    def plotter(self): return plotter.plot_frontface

class Perf_Curve(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_curve
    def plotter(self): return plotter.plot_perf_curve

class Perf_Excess(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess
    def plotter(self): return plotter.plot_perf_excess

class Drawdown(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_drawdown
    def plotter(self): return plotter.plot_perf_drawdown

class Excess_Drawdown(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_excess_drawdown
    def plotter(self): return plotter.plot_perf_excess_drawdown

class Perf_Year(BaseScreenPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_perf_year
    def plotter(self): return plotter.plot_perf_year
