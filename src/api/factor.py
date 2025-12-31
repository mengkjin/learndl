from typing import Any , Literal

from src.proj import Logger
from src.proj.util import Options
from src.data import DATAVENDOR
from src.res.factor.util import StockFactor
from src.res.factor.api import RiskModelUpdater , FactorCalculatorAPI , FactorTestAPI
from src.res.factor.calculator import StockFactorHierarchy

from .util import wrap_update

__all__ = [
    'FactorAPI' , 'get_random_factor' , 'get_real_factor' , 'get_factor' , 
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
               start_dt = 20240101 , end_dt = 20240331 , step = 5):
    if not names or names == 'random':
        Logger.stdout(f'Getting random factor values...')
        return get_random_factor(start_dt , end_dt , step)
    else:
        Logger.stdout(f'Getting factor values for {names}...')
        return get_real_factor(names , factor_type , start_dt , end_dt , step)

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

    class Affiliate:
        @classmethod
        def update(cls):
            # update risk factor
            wrap_update(FactorCalculatorAPI.Affiliate.update , 'update risk & external factors')

        @classmethod
        def rollback(cls , rollback_date : int):
            # update risk factor
            wrap_update(FactorCalculatorAPI.Affiliate.rollback , 'rollback risk & external factors' , rollback_date = rollback_date)
    
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
                       write_down = True , display_figs = False , indent : int = 0 , vb_level : int = 1 , 
                       **kwargs):
            with Logger.ParagraphIII(' test factor performance '):
                factor = get_factor(names , factor_type , start_dt , end_dt , step)
                ret = FactorTestAPI.FactorPerf(factor , benchmark , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs , **kwargs)
            return ret
        
        @staticmethod
        def FmpOptim(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
                     benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                     start_dt = 20240101 , end_dt = 20240331 , step = 5 ,
                     write_down = True , display_figs = False , indent : int = 0 , vb_level : int = 1 , 
                     prob_type : Literal['linprog' , 'quadprog' , 'socp'] = 'linprog' ,
                     **kwargs):
            with Logger.ParagraphIII(' test optimized fmp '):
                factor = get_factor(names , factor_type , start_dt , end_dt , step)
                ret = FactorTestAPI.FmpOptim(factor , benchmark , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs , prob_type = prob_type ,**kwargs)
            return ret
        
        @staticmethod
        def FmpTop(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
                   benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                   start_dt = 20240101 , end_dt = 20240331 , step = 5 ,
                   write_down = True , display_figs = False , indent : int = 0 , vb_level : int = 1 , 
                   **kwargs):
            with Logger.ParagraphIII(' test top fmp '):
                factor = get_factor(names , factor_type , start_dt , end_dt , step)
                ret = FactorTestAPI.FmpTop(factor , benchmark , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs , **kwargs)
            return ret

    @classmethod
    def FastAnalyze(cls , factor_name : str , start : int = 20170101 , end : int | None = None , step : int = 10 , lag = 2):
        factor_calc = cls.Hierarchy.get_factor(factor_name)
        dates = factor_calc.FactorDates(start,end,step)
        factor = factor_calc.Factor(dates , normalize = True , ignore_error = False)
        factor.fast_analyze(nday = step , lag = lag)
        return factor
    
    @classmethod
    def FullAnalyze(cls , factor_name : str , start : int = 20170101 , end : int | None = None , step : int = 1 , lag = 2):
        factor_calc = cls.Hierarchy.get_factor(factor_name)
        dates = factor_calc.FactorDates(start,end,step)
        factor = factor_calc.Factor(dates , normalize = True , ignore_error = False)
        factor.full_analyze(nday = step , lag = lag)
        return factor

    @classmethod
    def resume_testing_factors(cls):
        from src.api import ModelAPI
        for factor in Options.available_factors():
            with Logger.ParagraphI(f'Resume Testing Factor {factor}'):
                ModelAPI.test_factor(factor , resume = 1)