import pandas as pd

from . import stat as Stat
from . import plot as Plot
from ..test_manager import BaseCalculator

class BaseTopPortCalc(BaseCalculator):
    TASK_TYPE = 'top'
    DEFAULT_BENCHMARKS = 'defaults'
    def calc(self , account : pd.DataFrame , verbosity = 0):
        with self.suppress_warnings(): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        if verbosity > 0: print(f'    --->{self.__class__.__name__} calc Finished!')
        return self
    
class Top_FrontFace(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_frontface
    def plotter(self): return Plot.plot_top_frontface

class Top_Perf_Curve(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_curve
    def plotter(self): return Plot.plot_top_perf_curve

class Top_Perf_Excess(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_excess
    def plotter(self): return Plot.plot_top_perf_excess

class Top_Perf_Drawdown(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_drawdown
    def plotter(self): return Plot.plot_top_perf_drawdown

class Top_Perf_Excess_Drawdown(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_excess_drawdown
    def plotter(self): return Plot.plot_top_perf_excess_drawdown

class Top_Perf_Year(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_year
    def plotter(self): return Plot.plot_top_perf_year

class Top_Perf_Month(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_perf_month
    def plotter(self): return Plot.plot_top_perf_month

class Top_Exp_Style(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_exp_style
    def plotter(self): return Plot.plot_top_exp_style

class Top_Exp_Indus(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_exp_indus
    def plotter(self): return Plot.plot_top_exp_indus

class Top_Attrib_Source(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_attrib_source
    def plotter(self): return Plot.plot_top_attrib_source

class Top_Attrib_Style(BaseTopPortCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_top_attrib_style
    def plotter(self): return Plot.plot_top_attrib_style