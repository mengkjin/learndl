
import os , warnings
import numpy as np
import pandas as pd

from abc import abstractmethod , ABC
from dataclasses import asdict , dataclass , field
from matplotlib.figure import Figure
from typing import Any , Callable , Literal , Optional

from src.data import DataBlock

from typing import Any , Literal , Optional
from src.data import DataBlock

from ..basic import Benchmark , BENCHMARKS
from . import util as PerfUtil
from . import plot as PlotUtil

@dataclass
class PerfManager:
    perf_params : dict[str,Any] = field(default_factory=dict)
    all : bool = False
    ic_curve : bool = False
    ic_decay : bool = False
    ic_indus : bool = False
    ic_yearly : bool = False
    pnl_curve : bool = False
    style_corr : bool = False
    group_curve : bool = False
    group_decay_ret : bool = False
    group_decay_ir : bool = False
    group_yearly : bool = False
    distr_curve : bool = False
    distr_qtile : bool = False
    
    def __post_init__(self) -> None:
        self.perf_calc_dict = {k:self.select_perf_calc(k,self.perf_params) for k,v in self.perf_calc_bool.items() if v}

    def calc(self , factor_val: DataBlock | pd.DataFrame, benchmarks: Optional[list[Benchmark|Any]] = None):
        for _ , perf_calc in self.perf_calc_dict.items(): perf_calc.calc(factor_val , benchmarks)
        return self

    def plot(self , show = False):
        for _ , perf_calc in self.perf_calc_dict.items(): perf_calc.plot(show = show)
        return self
    
    def save(self , path : str):
        for perf_name , perf_calc in self.perf_calc_dict.items(): perf_calc.save(path = path , key = perf_name)
        return self

    @property
    def perf_calc_bool(self) -> dict[str,bool]:
        return {k:bool(v) or self.all for k,v in asdict(self).items() if k not in ['perf_params' , 'all']}

    @staticmethod
    def select_perf_calc(key , param) -> 'BasePerfCalc':
        return {
            'ic_curve' : ICCurve , 
            'ic_decay' : ICDecay ,
            'ic_indus' : ICIndustry ,
            'ic_yearly' : ICYearly ,
            'pnl_curve' : PnLCurve ,
            'style_corr' : StyleCorr ,
            'group_curve' : GroupCurve ,
            'group_decay_ret' : GroupDecayRet ,
            'group_decay_ir' :  GroupDecayIR ,
            'group_yearly' : GroupYearlyTop ,
            'distr_curve' : DistributionCurve ,
            'distr_qtile' : DistributionQuantile ,
        }[key](**param)
    
class suppress_warnings:
    def __enter__(self):
        warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='divide by zero encountered in divide', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='invalid value encountered in multiply', category=RuntimeWarning)
    def __exit__(self , *args):
        warnings.resetwarnings()

class BasePerfCalc(ABC):
    def __init__(self , **kwargs) -> None:
        self.params : dict[str,Any] = kwargs
        self.default_benchmarks : list[Benchmark|Any] = [None]
    @abstractmethod
    def calculator(self) -> Callable[...,pd.DataFrame]: '''Define calculator'''
    @abstractmethod
    def plotter(self) -> Callable: '''Define plotter'''
    def calc(self , factor_val : DataBlock | pd.DataFrame, benchmarks : Optional[list[Benchmark|Any]] | Any = None):
        with suppress_warnings(): 
            benchmarks = benchmarks if benchmarks is not None else self.default_benchmarks
            if not isinstance(benchmarks , list): benchmarks = [benchmarks]
            func = self.calculator()
            #self.benchmark_names = [(bm.name if bm else None) for bm in benchmarks]
            self.calc_rslt : pd.DataFrame = pd.concat(
                [func(factor_val,benchmark=bm,**self.params).assign(benchmark=(bm.name if bm else 'default')) for bm in benchmarks])
        return self
    def plot(self , show = False): 
        self.fig : Figure | dict[str,Figure] = self.plotter()(self.calc_rslt , show = show) #  benchmark = self.benchmark_names
        return self
    def save(self , path : str , key : Optional[str] = None):
        os.makedirs(path , exist_ok=True)
        if key is None: key = self.__class__.__name__
        self.calc_rslt.to_csv(os.path.join(path , f'{key}.csv'))
        if isinstance(self.fig , dict):
            [fig.savefig(os.path.join(path , f'{key}.{fac_name}.png')) for fac_name , fig in self.fig.items()]
        else:
            self.fig.savefig(os.path.join(path , f'{key}.png'))
        return self
    
class DistributionCurve(BasePerfCalc):
    def __init__(self , sampling_date_num : int = 12 , hist_bins : int = 50 , **kwargs) -> None:
        super().__init__(sampling_date_num = sampling_date_num , hist_bins = hist_bins)
        self.default_benchmarks = [None , *BENCHMARKS.values()]
    def calculator(self): return PerfUtil.calc_distribution
    def plotter(self): return PlotUtil.plot_distribution

class DistributionQuantile(BasePerfCalc):
    def __init__(self , scaling : bool = True , **kwargs) -> None:
        super().__init__(scaling = scaling)
        self.default_benchmarks = [None , *BENCHMARKS.values()]
    def calculator(self): return PerfUtil.calc_factor_qtile
    def plotter(self): return PlotUtil.plot_factor_qtile

class GroupCurve(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , group_num = group_num , ret_type = ret_type)
        self.default_benchmarks = [None , *BENCHMARKS.values()]
    def calculator(self): return PerfUtil.calc_grp_perf
    def plotter(self): return PlotUtil.plot_grp_perf

class GroupDecayRet(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag_init = lag_init , group_num = group_num , lag_num = lag_num , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_decay_grp_perf
    def plotter(self): return PlotUtil.plot_decay_grp_perf_ret

class GroupDecayIR(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag_init = lag_init , group_num = group_num , lag_num = lag_num , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_decay_grp_perf
    def plotter(self): return PlotUtil.plot_decay_grp_perf_ir

class GroupYearlyTop(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , group_num = group_num , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_top_grp_perf_yearly
    def plotter(self): return PlotUtil.plot_top_grp_perf_year

class ICDecay(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , lag_num : int = 5 ,
                 ic_type : Literal['pearson' , 'spearman'] = 'pearson' , 
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag_init = lag_init , lag_num = lag_num ,
                         ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_decay_ic
    def plotter(self): return PlotUtil.plot_decay_ic

class ICYearly(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_ic_year
    def plotter(self): return PlotUtil.plot_ic_year

class ICCurve(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , ma_windows : int | list[int] = [10,20] ,
                 ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , ma_windows = ma_windows ,ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_ic_curve
    def plotter(self): return PlotUtil.plot_ic_curve

class ICIndustry(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , 
                 ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_industry_ic
    def plotter(self): return PlotUtil.plot_industry_ic

class PnLCurve(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , given_direction : Literal[1,0,-1] = 0 ,
                 weight_type_list : list[str] = ['long' , 'long_short' , 'short'] , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , group_num = group_num , ret_type = ret_type ,
                         given_direction = given_direction , weight_type_list = weight_type_list)
        self.default_benchmarks = [None , *BENCHMARKS.values()]
    def calculator(self): return PerfUtil.calc_pnl
    def plotter(self): return PlotUtil.plot_pnl

class StyleCorr(BasePerfCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return PerfUtil.calc_style_corr
    def plotter(self): return PlotUtil.plot_style_corr

class StyleCorrBox(BasePerfCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return PerfUtil.calc_style_corr
    def plotter(self): return PlotUtil.plot_style_corr_box