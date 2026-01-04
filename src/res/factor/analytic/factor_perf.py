import pandas as pd

from typing import Any ,Literal , Type

from src.proj import Logger
from src.res.factor.util import Benchmark , StockFactor
from src.res.factor.util.plot.factor import Plotter
from src.res.factor.util.stat import factor as Stat

from .test_basics import BaseFactorAnalyticCalculator , BaseFactorAnalyticTest , test_title

test_type = 'factor'
plotter = Plotter(test_title(test_type))

class FactorPerfCalc(BaseFactorAnalyticCalculator):
    TEST_TYPE = test_type
    DEFAULT_BENCHMARKS : list[Benchmark|Any] | Benchmark | Any = [None]
    COMPULSORY_BENCHMARKS : Any = None
        
    def calc(self , factor : StockFactor, benchmarks : list[Benchmark|Any] | Any = None , indent : int = 1 , vb_level : int = 1):
        with self.calc_manager(f'{self.__class__.__name__} calc' , indent = indent , vb_level = vb_level):
            func = self.calculator()
            rslt = pd.concat([func(factor , bm , **self.params).assign(benchmark = bm.name) for bm in self.use_benchmarks(benchmarks)])
            self.calc_rslt = rslt.assign(benchmark = Benchmark.as_category(rslt['benchmark'])).set_index(['factor_name', 'benchmark']).sort_index()
        return self
    
    def use_benchmarks(self , benchmarks : list[Benchmark|Any] | Any = None):
        if self.COMPULSORY_BENCHMARKS is None:
            benchmarks = Benchmark.get_benchmarks(benchmarks if benchmarks is not None else self.DEFAULT_BENCHMARKS)
        else:
            benchmarks = Benchmark.get_benchmarks(self.COMPULSORY_BENCHMARKS)
        return benchmarks
    
class FrontFace(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , ic_type : Literal['pearson' , 'spearman'] = 'spearman' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type} , **kwargs)
    def calculator(self): return Stat.calc_frontface
    def plotter(self): return plotter.plot_frontface

class Coverage(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_coverage
    def plotter(self): return plotter.plot_coverage

class IC_Curve(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , ma_windows : int | list[int] = [10,20] ,
                 ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {
            'nday' : nday , 'lag' : lag , 'ma_windows' : ma_windows , 
            'ic_type' : ic_type , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_curve
    def plotter(self): return plotter.plot_ic_curve

class IC_Decay(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag_init : int = 2 , lag_num : int = 5 ,
                 ic_type : Literal['pearson' , 'spearman'] = 'spearman' , 
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {
            'nday' : nday , 'lag_init' : lag_init , 'lag_num' : lag_num , 
            'ic_type' : ic_type , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_decay
    def plotter(self): return plotter.plot_ic_decay

class IC_Indus(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , nday : int = 10 , lag : int = 2 , 
                 ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_indus
    def plotter(self): return plotter.plot_ic_indus

class IC_Year(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_year
    def plotter(self): return plotter.plot_ic_year

class IC_Benchmark(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = Benchmark.AVAILABLES
    def __init__(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_benchmark
    def plotter(self): return plotter.plot_ic_benchmark

class IC_Monotony(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , nday : int = 10 , lag_init : int = 2 , 
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag_init' : lag_init , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_monotony
    def plotter(self): return plotter.plot_ic_monotony

class PnL_Curve(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , direction : Literal[1,0,-1] = 0 , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'ret_type' : ret_type ,
                  'direction' : direction}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_pnl_curve
    def plotter(self): return plotter.plot_pnl_curve

class Style_Corr(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_style_corr
    def plotter(self): return plotter.plot_style_corr

class Style_Corr_Distrib(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_style_corr
    def plotter(self): return plotter.plot_style_corr_distrib

class Group_Return(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 20 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_group_return
    def plotter(self): return plotter.plot_group_return

class Group_Curve(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_group_curve
    def plotter(self): return plotter.plot_group_curve

class Group_Decay(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag_init' : lag_init , 'group_num' : group_num , 
                  'lag_num' : lag_num , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_group_decay
    def plotter(self): return plotter.plot_group_decay

class Group_IR_Decay(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag_init' : lag_init , 'group_num' : group_num , 
                  'lag_num' : lag_num , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_group_decay
    def plotter(self): return plotter.plot_group_ir_decay

class Group_Year(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_group_year
    def plotter(self): return plotter.plot_group_year

class Distrib_Curve(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , sampling_date_num : int = 12 , hist_bins : int = 50 , **kwargs) -> None:
        params = {'sampling_date_num' : sampling_date_num , 'hist_bins' : hist_bins}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_distrib_curve
    def plotter(self): return plotter.plot_distrib_curve

class Distrib_Qtile(FactorPerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , scaling : bool = True , **kwargs) -> None:
        params = {'scaling' : scaling}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_distrib_qtile
    def plotter(self): return plotter.plot_distrib_qtile

class FactorPerfTest(BaseFactorAnalyticTest):
    '''
    Factor Performance Calculator Manager
    Parameters:
        which : str | list[str] | Literal['all']
            Which tasks to run. Can be any of the following:
            'frontface' : Factor Front Face
            'coverage' : Factor Coverage
            'ic_curve' : IC Cumulative Curve
            'ic_decay' : IC Decay
            'ic_indus' : IC Industry
            'ic_year' : IC Year Stats
            'ic_benchmark' : IC Benchmark
            'ic_mono' : IC Monotony
            'pnl_curve' : PnL Cumulative Curve
            'style_corr' : Factor Style Correlation
            'grp_curve' : Group Return Cumulative Curve
            'grp_decay_ir' : Group Return Decay
            'grp_year' : Group Return Yearly Top
            'distr_curve' : Distribution Curve
        deprecated : 
            'grp_decay_ret' : Group Return Decay
            'distr_qtile' : Distribution Quantile
    '''
    TEST_TYPE = test_type
    TASK_LIST : list[Type[FactorPerfCalc]] = [
        FrontFace ,
        # Coverage ,
        IC_Curve , 
        # IC_Decay ,
        IC_Indus ,
        IC_Year ,
        IC_Benchmark ,
        IC_Monotony ,
        Group_Return ,
        # PnL_Curve ,
        # Style_Corr ,
        # Style_Corr_Distrib ,
        # Group_Curve ,
        # Group_Decay ,
        # Group_IR_Decay ,
        # Group_Year ,
        # Distrib_Curve ,
        # Distrib_Qtile ,
    ]

    def calc(self , factor: StockFactor , benchmarks: list[Benchmark|Any] | Any = None , indent : int = 0 , vb_level : int = 1):
        factor = factor.filter_dates_between(self.start_dt , self.end_dt)
        factor.cache_factor_stats.load(self.factor_stats_resume_path)
        with Logger.Timer(f'{self.__class__.__name__}.calc' , indent = indent , vb_level = vb_level , enter_vb_level = vb_level + 1):
            for task in self.tasks.values(): 
                task.calc(factor , benchmarks , indent = indent + 1 , vb_level = vb_level + 1)
        factor.cache_factor_stats.save(self.factor_stats_resume_path)
        return self
