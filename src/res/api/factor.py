import datetime
from typing import Any , Literal

from src.proj import Logger
from src.data import DATAVENDOR
from src.res.factor.util import StockFactor
from src.res.factor.api import FactorModelUpdater , FactorCalculatorAPI , FactorTestAPI
from src.res.factor.calculator import StockFactorHierarchy

__all__ = [
    'FactorAPI' , 'get_random_factor' , 'get_real_factor' , 'get_factor' , 'get_project_name' ,
    'FactorModelUpdater' , 'FactorCalculatorAPI' , 'FactorTestAPI' , 'StockFactorHierarchy' ,
    'StockFactor' , 'DATAVENDOR'
]

def get_random_factor(start_dt = 20240101 , end_dt = 20240331 , step = 5 , default_random_n = 2):
    return StockFactor(DATAVENDOR.random_factor(start_dt , end_dt , step , default_random_n).to_dataframe())

def get_real_factor(names = None ,
                    factor_type : Literal['factor' , 'pred'] = 'factor' , 
                    start_dt = 20240101 , end_dt = 20240331 , step = 5):
    assert names and names != 'random' , 'Names are required and not random for real factor!'
    return StockFactor(DATAVENDOR.real_factor(factor_type , names , start_dt , end_dt , step).to_dataframe())

def get_factor(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
               start_dt = 20240101 , end_dt = 20240331 , step = 5 , verbosity = 1):
    if not names or names == 'random':
        if verbosity > 0: 
            Logger.print(f'Getting random factor values...')
        return get_random_factor(start_dt , end_dt , step)
    else:
        if verbosity > 0: 
            Logger.print(f'Getting factor values for {names}...')
        return get_real_factor(names , factor_type , start_dt , end_dt , step)
    
def get_project_name(names = None , factor_type : Literal['factor' , 'pred'] = 'factor'):
    if not names or names == 'random':
        return 'random_factor'
    else:
        return f'{factor_type}_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

class FactorAPI:
    '''
    Interface for factor related operations
    .Test.FactorPerf()    : test factor performance
    .Test.FmpOptim()      : test optimized fmp
    .Test.FmpTop()        : test top fmp
    .factor_hierarchy()   : get factor hierarchy
    '''
    class Test:
        @staticmethod
        def FactorPerf(names = None ,
                       factor_type : Literal['factor' , 'pred'] = 'factor' , 
                       benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                       start_dt = 20240101 , end_dt = 20240331 , step = 5 ,
                       write_down = True , display_figs = False , verbosity = 1 , 
                       **kwargs):
            with Logger.EnclosedMessage(' test factor performance '):
                project_name = get_project_name(names , factor_type)
                factor = get_factor(names , factor_type , start_dt , end_dt , step , verbosity = verbosity)
                ret = FactorTestAPI.FactorPerf(factor , benchmark , write_down , display_figs , verbosity , project_name , **kwargs)
            return ret
        
        @staticmethod
        def FmpOptim(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
                     benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                     start_dt = 20240101 , end_dt = 20240331 , step = 5 ,
                     write_down = True , display_figs = False , verbosity = 1 , 
                     prob_type : Literal['linprog' , 'quadprog' , 'socp'] = 'linprog' ,
                     **kwargs):
            with Logger.EnclosedMessage(' test optimized fmp '):
                project_name = get_project_name(names , factor_type)
                factor = get_factor(names , factor_type , start_dt , end_dt , step , verbosity = verbosity)
                ret = FactorTestAPI.FmpOptim(factor , benchmark , write_down , display_figs , verbosity , project_name , prob_type = prob_type ,**kwargs)
            return ret
        
        @staticmethod
        def FmpTop(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
                   benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                   start_dt = 20240101 , end_dt = 20240331 , step = 5 ,
                   write_down = True , display_figs = False , verbosity = 1 , 
                   **kwargs):
            with Logger.EnclosedMessage(' test top fmp '):
                project_name = get_project_name(names , factor_type)
                factor = get_factor(names , factor_type , start_dt , end_dt , step , verbosity = verbosity)
                ret = FactorTestAPI.FmpTop(factor , benchmark , write_down , display_figs , verbosity , project_name , **kwargs)
            return ret
    
    @classmethod
    def Hierarchy(cls):
        return StockFactorHierarchy()

    @classmethod
    def FastAnalyze(cls , factor_name : str , start : int = 20170101 , end : int | None = None , step : int = 10 , lag = 2):
        factor_calc = cls.Hierarchy().get_factor(factor_name)
        factor = factor_calc.Factor(start,end,step,verbose= True , normalize = True , ignore_error = False)
        factor.fast_analyze(lag = lag)
        return factor
    
    @classmethod
    def FullAnalyze(cls , factor_name : str , start : int = 20170101 , end : int | None = None , step : int = 10 , lag = 2):
        factor_calc = cls.Hierarchy().get_factor(factor_name)
        factor = factor_calc.Factor(start,end,step,verbose= True , normalize = True , ignore_error = False)
        factor.full_analyze(lag = lag)
        return factor
