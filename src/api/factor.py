import datetime
from typing import Any , Literal

from src.factor.calculator import StockFactorHierarchy
from src.factor.analytic import PerfManager , FmpManager
from src.basic import PATH
from src.data import DATAVENDOR

def _project_path(prefix : str):
    start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    return PATH.result.joinpath(f'{prefix}_{start_time}')

def get_factor_vals(names = None ,
              factor_type : Literal['factor' , 'pred'] = 'factor' , 
              start_dt = 20240101 , end_dt = 20240531 , step = 5 ,
              default_random_n = 2):
    if names is None:
        factor_val = DATAVENDOR.random_factor(start_dt , end_dt , step , default_random_n).to_dataframe()
    else:
        factor_val = DATAVENDOR.real_factor(factor_type , names , start_dt , end_dt , step).to_dataframe()
    return factor_val

def perf_test(names = None ,
              factor_type : Literal['factor' , 'pred'] = 'factor' , 
              benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
              start_dt = 20240101 , end_dt = 20240531 , step = 5 ,
              write_down = True , display_figs = False):
    project_path = _project_path('perf_test')
    factor_val = get_factor_vals(names , factor_type , start_dt , end_dt , step)

    pm = PerfManager.run_test(factor_val , benchmark)
    if write_down:   pm.write_down(project_path)
    if display_figs: pm.display_figs()
    
    return pm

def fmp_test(names = None ,
             factor_type : Literal['factor' , 'pred'] = 'factor' , 
             benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
             start_dt = 20240101 , end_dt = 20240531 , step = 5 ,
             write_down = True , display_figs = False):
    project_path = _project_path('fmp_test')
    factor_val = get_factor_vals(names , factor_type , start_dt , end_dt , step)
    pm = FmpManager.run_test(factor_val , benchmark , verbosity=2)

    if write_down:   pm.write_down(project_path)
    if display_figs: pm.display_figs()
    return pm

def factor_hierarchy():
    return StockFactorHierarchy()

