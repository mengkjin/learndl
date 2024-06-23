import os , warnings
import pandas as pd

from abc import abstractmethod , ABC
from matplotlib.figure import Figure
from typing import Any , Callable , Literal , Optional

from . import stat as Stat
from . import plot as Plot
from ..util import Benchmark , BENCHMARKS
from ...data import DataBlock

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
        figs = self.plotter()(self.calc_rslt , show = show) #  benchmark = self.benchmark_names
        if isinstance(figs , Figure): figs = {'all':figs}
        self.figs : dict[str,Figure] = figs #  benchmark = self.benchmark_names
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
    
class Distribution_Curve(BasePerfCalc):
    def __init__(self , sampling_date_num : int = 12 , hist_bins : int = 50 , **kwargs) -> None:
        super().__init__(sampling_date_num = sampling_date_num , hist_bins = hist_bins)
    def calculator(self): return Stat.calc_distribution
    def plotter(self): return Plot.plot_distribution

class Distribution_Quantile(BasePerfCalc):
    def __init__(self , scaling : bool = True , **kwargs) -> None:
        super().__init__(scaling = scaling)
    def calculator(self): return Stat.calc_factor_qtile
    def plotter(self): return Plot.plot_factor_qtile

class Group_Ret_Cum_Curve(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , group_num = group_num , ret_type = ret_type)
        self.default_benchmarks = [None , *BENCHMARKS.values()]
    def calculator(self): return Stat.calc_grp_perf
    def plotter(self): return Plot.plot_grp_perf

class Group_Ret_Decay(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag_init = lag_init , group_num = group_num , lag_num = lag_num , ret_type = ret_type)
    def calculator(self): return Stat.calc_decay_grp_perf
    def plotter(self): return Plot.plot_decay_grp_perf_ret

class GroupDecayIR(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag_init = lag_init , group_num = group_num , lag_num = lag_num , ret_type = ret_type)
    def calculator(self): return Stat.calc_decay_grp_perf
    def plotter(self): return Plot.plot_decay_grp_perf_ir

class IC_Monotony(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , 
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag_init = lag_init , ret_type = ret_type)
    def calculator(self): return Stat.calc_ic_monotony
    def plotter(self): return Plot.plot_ic_monotony

class GroupYearTop(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , group_num = group_num , ret_type = ret_type)
    def calculator(self): return Stat.calc_top_grp_perf_year
    def plotter(self): return Plot.plot_top_grp_perf_year

class IC_Decay(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , lag_num : int = 5 ,
                 ic_type : Literal['pearson' , 'spearman'] = 'pearson' , 
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag_init = lag_init , lag_num = lag_num ,
                         ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return Stat.calc_decay_ic
    def plotter(self): return Plot.plot_decay_ic

class IC_Year_Stats(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return Stat.calc_ic_year
    def plotter(self): return Plot.plot_ic_year

class IC_Cum_Curve(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , ma_windows : int | list[int] = [10,20] ,
                 ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , ma_windows = ma_windows ,ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return Stat.calc_ic_curve
    def plotter(self): return Plot.plot_ic_curve

class IC_Industry(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , 
                 ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return Stat.calc_industry_ic
    def plotter(self): return Plot.plot_industry_ic

class PnL_Cum_Curve(BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , given_direction : Literal[1,0,-1] = 0 ,
                 weight_type_list : list[str] = ['long' , 'long_short' , 'short'] , **kwargs) -> None:
        super().__init__(nday = nday , lag = lag , group_num = group_num , ret_type = ret_type ,
                         given_direction = given_direction , weight_type_list = weight_type_list)
        self.default_benchmarks = [None , *BENCHMARKS.values()]
    def calculator(self): return Stat.calc_pnl
    def plotter(self): return Plot.plot_pnl

class Factor_Style_Corr(BasePerfCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_style_corr
    def plotter(self): return Plot.plot_style_corr

class Factor_Style_Corr_Distribution(BasePerfCalc):
    def __init__(self , **kwargs) -> None:
        super().__init__()
    def calculator(self): return Stat.calc_style_corr
    def plotter(self): return Plot.plot_style_corr_box