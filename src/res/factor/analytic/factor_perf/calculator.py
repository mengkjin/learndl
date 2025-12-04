import pandas as pd

from typing import Any , Literal

from ..test_manager import BaseCalculator
from src.res.factor.util import Benchmark , StockFactor
from src.res.factor.util.plot.factor import Plotter
from src.res.factor.util.stat import factor as Stat

plotter = Plotter('Factor')

class BasePerfCalc(BaseCalculator):
    TASK_TYPE = 'factor'
    DEFAULT_BENCHMARKS : list[Benchmark|Any] | Benchmark | Any = [None]
    DEFAULT_TITLE_GROUP = 'Factor'
    COMPULSORY_BENCHMARKS : Any = None
        
    def calc(self , factor : StockFactor, benchmarks : list[Benchmark|Any] | Any = None , verbosity = 0):
        with self.calc_manager(f'    --->{self.__class__.__name__} calc' , verbosity = verbosity):
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
    
class Factor_FrontFace(BasePerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , ic_type : Literal['pearson' , 'spearman'] = 'spearman' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        super().__init__(params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type} , **kwargs)
    def calculator(self): return Stat.calc_frontface
    def plotter(self): return plotter.plot_frontface

class Factor_Coverage(BasePerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_coverage
    def plotter(self): return plotter.plot_coverage

class Factor_IC_Curve(BasePerfCalc):
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

class Factor_IC_Decay(BasePerfCalc):
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

class Factor_IC_Indus(BasePerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , nday : int = 10 , lag : int = 2 , 
                 ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_indus
    def plotter(self): return plotter.plot_ic_indus

class Factor_IC_Year(BasePerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_year
    def plotter(self): return plotter.plot_ic_year

class Factor_IC_Benchmark(BasePerfCalc):
    COMPULSORY_BENCHMARKS = Benchmark.AVAILABLES
    def __init__(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_benchmark
    def plotter(self): return plotter.plot_ic_benchmark

class Factor_IC_Monotony(BasePerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , nday : int = 10 , lag_init : int = 2 , 
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag_init' : lag_init , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_ic_monotony
    def plotter(self): return plotter.plot_ic_monotony

class Factor_PnL_Curve(BasePerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , given_direction : Literal[1,0,-1] = 0 ,
                 weight_type_list : list[str] = ['long' , 'long_short' , 'short'] , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'ret_type' : ret_type ,
                  'given_direction' : given_direction , 'weight_type_list' : weight_type_list}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_pnl_curve
    def plotter(self): return plotter.plot_pnl_curve

class Factor_Style_Corr(BasePerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_style_corr
    def plotter(self): return plotter.plot_style_corr

class Factor_Style_Corr_Distrib(BasePerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , **kwargs) -> None:
        super().__init__(params = {} , **kwargs)
    def calculator(self): return Stat.calc_style_corr
    def plotter(self): return plotter.plot_style_corr_distrib

class Factor_Group_Curve(BasePerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_group_curve
    def plotter(self): return plotter.plot_group_curve

class Factor_Group_Decay(BasePerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag_init' : lag_init , 'group_num' : group_num , 
                  'lag_num' : lag_num , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_group_decay
    def plotter(self): return plotter.plot_group_decay

class Factor_Group_IR_Decay(BasePerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                 lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag_init' : lag_init , 'group_num' : group_num , 
                  'lag_num' : lag_num , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_group_decay
    def plotter(self): return plotter.plot_group_ir_decay

class Factor_Group_Year(BasePerfCalc):
    COMPULSORY_BENCHMARKS = ['market' , 'csi300' , 'csi500' , 'csi1000']
    def __init__(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                 ret_type : Literal['close' , 'vwap'] = 'close' , **kwargs) -> None:
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'ret_type' : ret_type}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_group_year
    def plotter(self): return plotter.plot_group_year

class Factor_Distrib_Curve(BasePerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , sampling_date_num : int = 12 , hist_bins : int = 50 , **kwargs) -> None:
        params = {'sampling_date_num' : sampling_date_num , 'hist_bins' : hist_bins}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_distrib_curve
    def plotter(self): return plotter.plot_distrib_curve

class Factor_Distrib_Qtile(BasePerfCalc):
    COMPULSORY_BENCHMARKS = 'market'
    def __init__(self , scaling : bool = True , **kwargs) -> None:
        params = {'scaling' : scaling}
        super().__init__(params = params , **kwargs)
    def calculator(self): return Stat.calc_distrib_qtile
    def plotter(self): return plotter.plot_distrib_qtile