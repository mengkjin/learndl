import datetime
from typing import Any , Literal

from src.proj import Logger
from src.data import DATAVENDOR
from src.res.factor.util import StockFactor
from src.res.factor.api import RiskModelUpdater , FactorCalculatorAPI , FactorTestAPI
from src.res.factor.calculator import StockFactorHierarchy

from .util import wrap_update

__all__ = [
    'FactorAPI' , 'get_random_factor' , 'get_real_factor' , 'get_factor' , 'get_project_name' ,
    'RiskModelUpdater' , 'FactorCalculatorAPI' , 'FactorTestAPI' , 'StockFactorHierarchy' ,
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
    """
    Interface for factor related operations
    .update()             : update factor
    .Test.FactorPerf()    : test factor performance
    .Test.FmpOptim()      : test optimized fmp
    .Test.FmpTop()        : test top fmp
    .factor_hierarchy()   : get factor hierarchy
    """
    class Hierarchy(StockFactorHierarchy):
        @classmethod
        def update(cls):
            wrap_update(StockFactorHierarchy.export_factor_table , 'export factor table')

        @classmethod
        def rollback(cls , rollback_date : int = -1):
            wrap_update(StockFactorHierarchy.export_factor_table , 'export factor table')

    class Stock:
        @classmethod
        def update(cls , timeout : int = -1):
            # update stock factor
            wrap_update(FactorCalculatorAPI.Stock.update , 'update stock factors' , timeout = timeout)

        @classmethod
        def rollback(cls , rollback_date : int , timeout : int = -1):
            # update stock factor
            wrap_update(FactorCalculatorAPI.Stock.rollback , 'rollback stock factors' , rollback_date = rollback_date , timeout = timeout)

    class Market:
        @classmethod
        def update(cls):
            # update market factor
            wrap_update(FactorCalculatorAPI.Market.update , 'update market factors')

        @classmethod
        def rollback(cls , rollback_date : int):
            # update market factor
            wrap_update(FactorCalculatorAPI.Market.rollback , 'rollback market factors' , rollback_date = rollback_date)

    class Risk:
        @classmethod
        def update(cls):
            # update risk factor
            wrap_update(FactorCalculatorAPI.Risk.update , 'update risk factors')

        @classmethod
        def rollback(cls , rollback_date : int):
            # update risk factor
            wrap_update(FactorCalculatorAPI.Risk.rollback , 'rollback risk factors' , rollback_date = rollback_date)
    
    class Pooling:
        @classmethod
        def update(cls , timeout : int = -1):
            # update pooling factor
            wrap_update(FactorCalculatorAPI.Pooling.update , 'update pooling factors' , timeout = timeout)

        @classmethod
        def rollback(cls , rollback_date : int , timeout : int = -1):
            # update pooling factor
            wrap_update(FactorCalculatorAPI.Pooling.rollback , 'rollback pooling factors' , rollback_date = rollback_date , timeout = timeout)

    class Stats:
        @classmethod
        def update(cls):
            # update factor stats
            wrap_update(FactorCalculatorAPI.Stats.update , 'update factor stats')

        @classmethod
        def rollback(cls , rollback_date : int):
            # update factor stats
            wrap_update(FactorCalculatorAPI.Stats.rollback , 'rollback factor stats' , rollback_date = rollback_date)

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
    def FastAnalyze(cls , factor_name : str , start : int = 20170101 , end : int | None = None , step : int = 10 , lag = 2):
        factor_calc = cls.Hierarchy.get_factor(factor_name)
        dates = factor_calc.FactorDates(start,end,step)
        factor = factor_calc.Factor(dates , verbose= True , normalize = True , ignore_error = False)
        factor.fast_analyze(nday = step , lag = lag)
        return factor
    
    @classmethod
    def FullAnalyze(cls , factor_name : str , start : int = 20170101 , end : int | None = None , step : int = 1 , lag = 2):
        factor_calc = cls.Hierarchy.get_factor(factor_name)
        dates = factor_calc.FactorDates(start,end,step)
        factor = factor_calc.Factor(dates , verbose= True , normalize = True , ignore_error = False)
        factor.full_analyze(nday = step , lag = lag)
        return factor