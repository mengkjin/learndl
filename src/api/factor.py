import datetime
from typing import Any , Literal

from src.factor.calculator import StockFactorHierarchy
from src.factor.analytic import FactorPerfManager , FmpOptimManager , FmpTopManager , FactorTestAPI
from src.basic import PATH
from src.data import DATAVENDOR

def get_random_factor(start_dt = 20240101 , end_dt = 20240331 , step = 5 , default_random_n = 2):
    return DATAVENDOR.random_factor(start_dt , end_dt , step , default_random_n).to_dataframe()

def get_real_factor(names = None ,
                    factor_type : Literal['factor' , 'pred'] = 'factor' , 
                    start_dt = 20240101 , end_dt = 20240331 , step = 5):
    assert names is not None , 'Names are required for real factor!'
    return DATAVENDOR.real_factor(factor_type , names , start_dt , end_dt , step).to_dataframe()

def get_factor_vals(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
                     start_dt = 20240101 , end_dt = 20240331 , step = 5 , verbosity = 1):
    
    if names is None:
        if verbosity > 0: print(f'Getting random factor values...')
        return DATAVENDOR.random_factor(start_dt , end_dt , step).to_dataframe()
    else:
        if verbosity > 0: print(f'Getting factor values for {names}...')
        return DATAVENDOR.real_factor(factor_type , names , start_dt , end_dt , step).to_dataframe()

class FactorAPI:
    class Test:
        @staticmethod
        def FactorPerf(names = None ,
                       factor_type : Literal['factor' , 'pred'] = 'factor' , 
                       benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                       start_dt = 20240101 , end_dt = 20240331 , step = 5 ,
                       write_down = True , display_figs = False , verbosity = 1 , **kwargs):
            factor_val = get_factor_vals(names , factor_type , start_dt , end_dt , step , verbosity = verbosity)
            return FactorTestAPI.FactorPerf(factor_val , benchmark , write_down , display_figs , verbosity , **kwargs)
        
        @staticmethod
        def FmpOptim(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
                     benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                     start_dt = 20240101 , end_dt = 20240331 , step = 5 ,
                     write_down = True , display_figs = False , verbosity = 1 , **kwargs):
            factor_val = get_factor_vals(names , factor_type , start_dt , end_dt , step , verbosity = verbosity)
            return FactorTestAPI.FmpOptim(factor_val , benchmark , write_down , display_figs , verbosity , **kwargs)
        
        @staticmethod
        def FmpTop(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
                   benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                   start_dt = 20240101 , end_dt = 20240331 , step = 5 ,
                   write_down = True , display_figs = False , verbosity = 1 , **kwargs):
            factor_val = get_factor_vals(names , factor_type , start_dt , end_dt , step , verbosity = verbosity)
            return FactorTestAPI.FmpTop(factor_val , benchmark , write_down , display_figs , verbosity , **kwargs)
    
    @classmethod
    def factor_hierarchy(cls):
        return StockFactorHierarchy()
    