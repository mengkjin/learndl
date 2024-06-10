import pandas as pd
from typing import Literal 

from .data_util import DATAVENDOR
from .port_util import Benchmark
from . import perf_util as PerfUtil 
from . import plot_util as PlotUtil

__all__ = ['DistributionCurve' , 'DistributionQuantile' ,
           'GroupCurve' , 'GroupDecayRet' , 'GroupDecayIR' , 'GroupYearlyTop' ,
           'ICCurve' , 'ICDecay' , 'ICIndustry' , 'ICYearly' ,
           'PnLCurve' ,
           'StyleCorr']

class DistributionCurve(PerfUtil.BasePerfCalc):
    def __init__(self , sampling_date_num : int = 12 , hist_bins : int = 50) -> None:
        super().__init__(sampling_date_num = sampling_date_num , hist_bins = hist_bins)
    def calculator(self): return PerfUtil.calc_distribution
    def plotter(self): return PlotUtil.plot_distribution

class DistributionQuantile(PerfUtil.BasePerfCalc):
    def __init__(self , scaling : bool = True) -> None:
        super().__init__(scaling = scaling)
    def calculator(self): return PerfUtil.calc_factor_qtile
    def plotter(self): return PlotUtil.plot_factor_qtile

class GroupCurve(PerfUtil.BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close') -> None:
        super().__init__(nday = nday , lag = lag , group_num = group_num , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_grp_perf
    def plotter(self): return PlotUtil.plot_grp_perf

class GroupDecayRet(PerfUtil.BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close') -> None:
        super().__init__(nday = nday , lag_init = lag_init , group_num = group_num , lag_num = lag_num , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_decay_grp_perf
    def plotter(self): return PlotUtil.plot_decay_grp_perf
    def plot(self): self.plotter()(self.calc_rslt , stat_type = 'ret')

class GroupDecayIR(PerfUtil.BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close') -> None:
        super().__init__(nday = nday , lag_init = lag_init , group_num = group_num , lag_num = lag_num , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_decay_grp_perf
    def plotter(self): return PlotUtil.plot_decay_grp_perf
    def plot(self): self.plotter()(self.calc_rslt , stat_type = 'ir')

class GroupYearlyTop(PerfUtil.BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close') -> None:
        super().__init__(nday = nday , lag = lag , group_num = group_num , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_top_grp_perf_yearly
    def plotter(self): return PlotUtil.plot_top_grp_perf_year

class ICDecay(PerfUtil.BasePerfCalc):
    def __init__(self , nday : int = 10 , lag_init : int = 2 , lag_num : int = 5 ,
                 ic_type : Literal['pearson' , 'spearman'] = 'pearson' , 
                 ret_type : Literal['close' , 'vwap'] = 'close') -> None:
        super().__init__(nday = nday , lag_init = lag_init , lag_num = lag_num ,
                         ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_decay_ic
    def plotter(self): return PlotUtil.plot_decay_ic

class ICYearly(PerfUtil.BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close') -> None:
        super().__init__(nday = nday , lag = lag , ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_ic_year
    def plotter(self): return PlotUtil.plot_ic_year

class ICCurve(PerfUtil.BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , ma_windows : int | list[int] = [10,20] ,
                 ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close') -> None:
        super().__init__(nday = nday , lag = lag , ma_windows = ma_windows ,ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_ic_curve
    def plotter(self): return PlotUtil.plot_ic_curve

class ICIndustry(PerfUtil.BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , 
                 ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close') -> None:
        super().__init__(nday = nday , lag = lag , ic_type = ic_type , ret_type = ret_type)
    def calculator(self): return PerfUtil.calc_industry_ic
    def plotter(self): return PlotUtil.plot_industry_ic

class PnLCurve(PerfUtil.BasePerfCalc):
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , given_direction : Literal[1,0,-1] = 0 ,
                 weight_type_list : list[str] = ['long' , 'long_short' , 'short']) -> None:
        super().__init__(nday = nday , lag = lag , group_num = group_num , ret_type = ret_type ,
                         given_direction = given_direction , weight_type_list = weight_type_list)
    def calculator(self): return PerfUtil.calc_pnl
    def plotter(self): return PlotUtil.plot_pnl

class StyleCorr(PerfUtil.BasePerfCalc):
    def __init__(self) -> None:
        super().__init__()
    def calculator(self): return PerfUtil.calc_style_corr
    def plotter(self): return PlotUtil.plot_style_corr_curve # PlotUtil.plot_style_corr

def main_test():
    factor_val = DATAVENDOR.random_factor(20230701 , 20240331).to_dataframe()
    benchmark  = Benchmark('csi500')

    del factor_val['factor2']
    print('test : ' , __all__)
    for calc_name in __all__:
        a = globals()[calc_name]()
        a.calc(factor_val , benchmark).plot()