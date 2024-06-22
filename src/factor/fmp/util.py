import itertools , time

import pandas as pd
import numpy as np

from dataclasses import dataclass , field
from typing import Any , Literal

from ..basic import DATAVENDOR , AlphaModel , RISK_MODEL , Portfolio , BENCHMARKS , Benchmark , Port
from ..basic.var import ROUNDING_RETURN , ROUNDING_TURNOVER
from ..optimizer.api import PortfolioOptimizer , PortOptimResult

    
import os , warnings
import pandas as pd

from abc import abstractmethod , ABC
from matplotlib.figure import Figure
from typing import Any , Callable , Literal , Optional

from src.data import DataBlock

from ..basic import Benchmark , BENCHMARKS
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
    def save(self , path : str , key : Optional[str] = None):
        os.makedirs(path , exist_ok=True)
        if key is None: key = self.__class__.__name__
        self.calc_rslt.to_csv(os.path.join(path , f'{key}.csv'))
        [fig.savefig(os.path.join(path , f'{key}.{fig_name}.png')) for fig_name , fig in self.figs.items()]
        return self
    
class FmpPrefix(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_prefix
    def plotter(self): return Plot.plot_fmp_prefix

class FmpPerfCurve(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_perf_curve
    def plotter(self): return Plot.plot_fmp_perf_curve

class FmpLagCurve(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_lag_curve
    def plotter(self): return Plot.plot_fmp_lag_curve

class FmpPerfYear(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_perf_year
    def plotter(self): return Plot.plot_fmp_perf_year

class FmpPerfMonth(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_perf_month
    def plotter(self): return Plot.plot_fmp_perf_month

class FmpStyleExp(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_style_exp
    def plotter(self): return Plot.plot_fmp_style_exp

class FmpInustryExp(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_industry_exp
    def plotter(self): return Plot.plot_fmp_industry_exp

class FmpAttributionCurve(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_attrib_curve
    def plotter(self): return Plot.plot_fmp_attrib_curve

class FmpAttributionStyleCurve(BaseFmpCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_fmp_style_attrib_curve
    def plotter(self): return Plot.plot_fmp_style_attrib_curve