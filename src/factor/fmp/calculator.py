import os , warnings
import pandas as pd

from abc import abstractmethod , ABC
from matplotlib.figure import Figure
from typing import Any , Callable

from . import stat as Stat
from . import plot as Plot

class suppress_warnings:
    def __enter__(self):
        warnings.filterwarnings('ignore', message='divide by zero encountered in divide', category=RuntimeWarning)
    def __exit__(self , *args):
        warnings.resetwarnings()

class BaseFmpCalc(ABC):
    def __init__(self , **kwargs) -> None:
        self.params : dict[str,Any] = kwargs
    @abstractmethod
    def calculator(self) -> Callable[...,pd.DataFrame]: '''Define calculator'''
    @abstractmethod
    def plotter(self) -> Callable: '''Define plotter'''
    def calc(self , account : pd.DataFrame):
        with suppress_warnings(): 
            self.calc_rslt : pd.DataFrame = self.calculator()(account)
        return self
    def plot(self , show = False): 
        figs = self.plotter()(self.calc_rslt , show = show) #  benchmark = self.benchmark_names
        if isinstance(figs , Figure): figs = {'all':figs}
        self.figs : dict[str,Figure] = figs
        return self
    def save(self , path : str):
        os.makedirs(path , exist_ok=True)
        os.makedirs(os.path.join(path , 'data') , exist_ok=True)
        os.makedirs(os.path.join(path , 'plot') , exist_ok=True)
        sub_key = self.__class__.__name__
        self.calc_rslt.to_csv(os.path.join(path , 'data' ,f'{sub_key}.csv'))
        for fig_name , fig in self.figs.items():
            if fig_name == 'all':
                fig.savefig(os.path.join(path , 'plot' , f'{sub_key}.{fig_name}.png'))
            else:
                factor_name , bm_name = fig_name.split('.')
                os.makedirs(os.path.join(path , 'plot' , factor_name) , exist_ok=True)
                fig.savefig(os.path.join(path , 'plot' , factor_name ,  f'{sub_key}.{bm_name}.png'))
        return self
    
class Fmp_Prefix(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_prefix
    def plotter(self): return Plot.plot_fmp_prefix

class Fmp_Perf_Curve(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_perf_curve
    def plotter(self): return Plot.plot_fmp_perf_curve

class Fmp_Perf_Lag_Curve(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_perf_lag
    def plotter(self): return Plot.plot_fmp_perf_lag

class Fmp_Year_Stats(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_perf_year
    def plotter(self): return Plot.plot_fmp_perf_year

class Fmp_Month_Stats(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_perf_month
    def plotter(self): return Plot.plot_fmp_perf_month

class Fmp_Style_Exposure(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_style_exp
    def plotter(self): return Plot.plot_fmp_style_exp

class Fmp_Inustry_Deviation(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_industry_exp
    def plotter(self): return Plot.plot_fmp_industry_exp

class Fmp_Attribution_Source(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_attrib_source
    def plotter(self): return Plot.plot_fmp_attrib_source

class Fmp_Attribution_Style(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_attrib_style
    def plotter(self): return Plot.plot_fmp_attrib_style