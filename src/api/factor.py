from typing import Any , Literal

from src.proj import Logger , Proj
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

def get_random_factor(start = 20240101 , end = 20240331 , step = 5 , default_random_n = 2):
    """
    Build a ``StockFactor`` from a random synthetic vendor slice (debug/experiments).
    """
    return StockFactor(DATAVENDOR.random_factor(start , end , step , default_random_n).to_dataframe())

def get_real_factor(names = None ,
                    factor_type : Literal['factor' , 'pred'] = 'factor' , 
                    start = 20240101 , end = 20240331 , step = 5):
    """
    Load named real factors/preds from ``DATAVENDOR`` into a ``StockFactor``.
    """
    assert names and names != 'random' , 'Names are required and not random for real factor!'
    return StockFactor(DATAVENDOR.real_factor(factor_type , names , start , end , step).to_dataframe())

def get_factor(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
               start = 20240101 , end = 20240331 , step = 5):
    """
    Dispatch to ``get_random_factor`` or ``get_real_factor`` depending on *names*.
    """
    if not names or names == 'random':
        Logger.stdout(f'Getting random factor values...')
        return get_random_factor(start , end , step)
    else:
        Logger.stdout(f'Getting factor values for {names}...')
        return get_real_factor(names , factor_type , start , end , step)

class FactorAPI:
    """
    Interface for factor related operations
    .update()             : update factor
    .Test.FactorPerf()    : test factor performance
    .Test.FmpOptim()      : test optimized fmp
    .Test.FmpTop()        : test top fmp
    .factor_hierarchy()   : get factor hierarchy
    """
    @classmethod
    def export_factor_table(cls , return_path : bool = False):
        """
        Export / refresh the factor hierarchy table materialization.

        [API Interaction]:
            expose: false
            email: true
            roles: [user, developer, admin]
            risk: write
            lock_num: 1
            lock_timeout: 1
            disable_platforms: []
            execution_time: short
            memory_usage: medium
        """
        path = wrap_update(StockFactorHierarchy.export_factor_table , 'export factor table')
        if path is not None and path.exists() and return_path:
            return path
    class Stock:
        @classmethod
        def update(cls , timeout : int = -1):
            """
            Refresh stock factors via ``FactorCalculatorAPI.Stock``.

            Args:
                timeout: Calculator timeout hint (-1 uses defaults).

            [API Interaction]:
              expose: false
              roles: [developer, admin]
              risk: write
              lock_num: -1
              disable_platforms: []
              execution_time: long
              memory_usage: high
            """
            wrap_update(FactorCalculatorAPI.Stock.update , 'update stock factors' , timeout = timeout)

        @classmethod
        def rollback(cls , rollback_date : int , timeout : int = -1):
            """
            Roll back stock factors to *rollback_date*.

            [API Interaction]:
              expose: false
              roles: [admin]
              risk: destructive
              lock_num: -1
              disable_platforms: []
              execution_time: long
              memory_usage: high
            """
            wrap_update(FactorCalculatorAPI.Stock.rollback , 'rollback stock factors' , rollback_date = rollback_date , timeout = timeout)

    class Market:
        @classmethod
        def update(cls):
            """
            Refresh market-wide factors.

            [API Interaction]:
              expose: false
              roles: [developer, admin]
              risk: write
              lock_num: -1
              disable_platforms: []
              execution_time: medium
              memory_usage: high
            """
            wrap_update(FactorCalculatorAPI.Market.update , 'update market factors')

        @classmethod
        def rollback(cls , rollback_date : int):
            """
            Roll back market factors to *rollback_date*.

            [API Interaction]:
              expose: false
              roles: [admin]
              risk: destructive
              lock_num: -1
              disable_platforms: []
              execution_time: medium
              memory_usage: high
            """
            wrap_update(FactorCalculatorAPI.Market.rollback , 'rollback market factors' , rollback_date = rollback_date)

    class Affiliate:
        @classmethod
        def update(cls):
            """
            Refresh affiliate / external risk factors.

            [API Interaction]:
              expose: false
              roles: [developer, admin]
              risk: write
              lock_num: -1
              disable_platforms: []
              execution_time: medium
              memory_usage: high
            """
            wrap_update(FactorCalculatorAPI.Affiliate.update , 'update risk & external factors')

        @classmethod
        def rollback(cls , rollback_date : int):
            """
            Roll back affiliate factors to *rollback_date*.

            [API Interaction]:
              expose: false
              roles: [admin]
              risk: destructive
              lock_num: -1
              disable_platforms: []
              execution_time: medium
              memory_usage: high
            """
            wrap_update(FactorCalculatorAPI.Affiliate.rollback , 'rollback risk & external factors' , rollback_date = rollback_date)
    
    class Pooling:
        @classmethod
        def update(cls , timeout : int = -1):
            """
            Refresh pooling-layer factors.

            [API Interaction]:
              expose: false
              roles: [developer, admin]
              risk: write
              lock_num: -1
              disable_platforms: []
              execution_time: long
              memory_usage: high
            """
            wrap_update(FactorCalculatorAPI.Pooling.update , 'update pooling factors' , timeout = timeout)

        @classmethod
        def rollback(cls , rollback_date : int , timeout : int = -1):
            """
            Roll back pooling factors to *rollback_date*.

            [API Interaction]:
              expose: false
              roles: [admin]
              risk: destructive
              lock_num: -1
              disable_platforms: []
              execution_time: long
              memory_usage: high
            """
            wrap_update(FactorCalculatorAPI.Pooling.rollback , 'rollback pooling factors' , rollback_date = rollback_date , timeout = timeout)

    class Stats:
        @classmethod
        def update(cls):
            """
            Refresh factor statistics tables.

            [API Interaction]:
              expose: false
              roles: [developer, admin]
              risk: write
              lock_num: -1
              disable_platforms: []
              execution_time: medium
              memory_usage: medium
            """
            wrap_update(FactorCalculatorAPI.Stats.update , 'update factor stats')

        @classmethod
        def rollback(cls , rollback_date : int):
            """
            Roll back factor stats to *rollback_date*.

            [API Interaction]:
              expose: false
              roles: [admin]
              risk: destructive
              lock_num: -1
              disable_platforms: []
              execution_time: medium
              memory_usage: medium
            """
            wrap_update(FactorCalculatorAPI.Stats.rollback , 'rollback factor stats' , rollback_date = rollback_date)

    class Test:
        @staticmethod
        def FactorPerf(names = None ,
                       factor_type : Literal['factor' , 'pred'] = 'factor' , 
                       benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                       start = 20240101 , end = 20240331 , step = 5 ,
                       write_down = True , display_figs = False , indent : int = 0 , vb_level : Any = 1 , 
                       **kwargs):
            """
            Run ``FactorTestAPI.FactorPerf`` on factors resolved via ``get_factor``.

            [API Interaction]:
              expose: false
              roles: [developer, admin]
              risk: read_only
              lock_num: -1
              disable_platforms: []
              execution_time: medium
              memory_usage: high
            """
            with Logger.Paragraph('test factor performance' , 3):
                factor = get_factor(names , factor_type , start , end , step)
                ret = FactorTestAPI.FactorPerf(factor , benchmark , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs , **kwargs)
            return ret
        
        @staticmethod
        def FmpOptim(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
                     benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                     start = 20240101 , end = 20240331 , step = 5 ,
                     write_down = True , display_figs = False , indent : int = 0 , vb_level : Any = 1 , 
                     prob_type : Literal['linprog' , 'quadprog' , 'socp'] = 'linprog' ,
                     **kwargs):
            """
            Run optimized FMP diagnostics via ``FactorTestAPI.FmpOptim``.

            [API Interaction]:
              expose: false
              roles: [developer, admin]
              risk: read_only
              lock_num: -1
              disable_platforms: []
              execution_time: medium
              memory_usage: high
            """
            with Logger.Paragraph('test optimized fmp' , 3):
                factor = get_factor(names , factor_type , start , end , step)
                ret = FactorTestAPI.FmpOptim(factor , benchmark , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs , prob_type = prob_type ,**kwargs)
            return ret
        
        @staticmethod
        def FmpTop(names = None , factor_type : Literal['factor' , 'pred'] = 'factor' , 
                   benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                   start = 20240101 , end = 20240331 , step = 5 ,
                   write_down = True , display_figs = False , indent : int = 0 , vb_level : Any = 1 , 
                   **kwargs):
            """
            Run top-FMP diagnostics via ``FactorTestAPI.FmpTop``.

            [API Interaction]:
              expose: false
              roles: [developer, admin]
              risk: read_only
              lock_num: -1
              disable_platforms: []
              execution_time: medium
              memory_usage: high
            """
            with Logger.Paragraph('test top fmp' , 3):
                factor = get_factor(names , factor_type , start , end , step)
                ret = FactorTestAPI.FmpTop(factor , benchmark , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs , **kwargs)
            return ret

    @classmethod
    def FastAnalyze(cls , factor_name : str , start : int = 20170101 , end : int | None = None , step : int = 10 , lag = 2):
        """
        Coarse-step ``fast_analyze`` for *factor_name* over ``Hierarchy`` dates.

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: medium
          memory_usage: high
        """
        factor_calc = cls.Hierarchy.get_factor(factor_name)
        dates = factor_calc.FactorDates(start,end,step)
        factor = factor_calc.Factor(dates , normalize = True , ignore_error = False)
        factor.fast_analyze(nday = step , lag = lag)
        return factor
    
    @classmethod
    def FullAnalyze(cls , factor_name : str , start : int = 20170101 , end : int | None = None , step : int = 1 , lag = 2):
        """
        Fine-step ``full_analyze`` for *factor_name* over ``Hierarchy`` dates.

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        """
        factor_calc = cls.Hierarchy.get_factor(factor_name)
        dates = factor_calc.FactorDates(start,end,step)
        factor = factor_calc.Factor(dates , normalize = True , ignore_error = False)
        factor.full_analyze(nday = step , lag = lag)
        return factor

    @classmethod
    def resume_testing_factors(cls):
        """
        Resume ``ModelAPI.test_factor`` for every factor listed in ``Options.available_factors``.

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        """
        from src.api import ModelAPI
        for factor in Options.available_factors():
            title = f'Resume Testing Factor {factor}'
            with Logger.Paragraph(title , 1) if Proj.vb.vb > 1 else Logger.Timer(title):
                ModelAPI.test_factor(factor , resume = 1)