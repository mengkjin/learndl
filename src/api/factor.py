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

def get_real_factor(names : str | list[str] | None = None ,
                    factor_type : Literal['factor' , 'pred'] = 'factor' , 
                    start = 20240101 , end = 20240331 , step = 5):
    """
    Load named real factors/preds from ``DATAVENDOR`` into a ``StockFactor``.
    """
    assert names and names != 'random' , 'Names are required and not random for real factor!'
    return StockFactor(DATAVENDOR.real_factor(factor_type , names , start , end , step).to_dataframe())

def get_factor(names : str | list[str] | None = None , 
               factor_type : Literal['factor' , 'pred'] = 'factor' , 
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
    def export_factor_table(cls):
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
        if path is not None and path.exists():
            return path

    @classmethod
    def update_stock_factors(cls , timeout : int = -1):
        """
        Update stock factors via ``FactorCalculatorAPI.Stock``.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Stock.update , 'update stock factors' , timeout = timeout)

    @classmethod
    def rollback_stock_factors(cls , rollback_date : int , timeout : int = -1):
        """
        Rollback stock factors via ``FactorCalculatorAPI.Stock``.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Stock.rollback , 'rollback stock factors' , rollback_date = rollback_date , timeout = timeout)
    
    @classmethod
    def update_market_factors(cls):
        """
        Update market-wide factors via ``FactorCalculatorAPI.Market``.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Market.update , 'update market factors')

    @classmethod
    def rollback_market_factors(cls , rollback_date : int):
        """
        Rollback market-wide factors via ``FactorCalculatorAPI.Market``.

        [API Interaction]:
          expose: true
          email: true
          roles: [admin]
          risk: destructive
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Market.rollback , 'rollback market factors' , rollback_date = rollback_date)

    @classmethod
    def update_affiliate_factors(cls):
        """
        Update affiliate / external risk factors via ``FactorCalculatorAPI.Affiliate``.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Affiliate.update , 'update risk & external factors')

    @classmethod
    def rollback_affiliate_factors(cls , rollback_date : int):
        """
        Rollback affiliate / external risk factors via ``FactorCalculatorAPI.Affiliate``.

        [API Interaction]:
          expose: true
          email: true
          roles: [admin]
          risk: destructive
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Affiliate.rollback , 'rollback risk & external factors' , rollback_date = rollback_date)

    @classmethod
    def update_pooling_factors(cls , timeout : int = -1):
        """
        Update pooling-layer factors via ``FactorCalculatorAPI.Pooling``.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Pooling.update , 'update pooling factors' , timeout = timeout)

    @classmethod
    def rollback_pooling_factors(cls , rollback_date : int , timeout : int = -1):
        """
        Rollback pooling-layer factors via ``FactorCalculatorAPI.Pooling``.

        [API Interaction]:
          expose: true
          email: true
          roles: [admin]
          risk: destructive
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Pooling.rollback , 'rollback pooling factors' , rollback_date = rollback_date , timeout = timeout)

    @classmethod
    def update_factor_stats(cls):
        """
        Update factor statistics tables via ``FactorCalculatorAPI.Stats``.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Stats.update , 'update factor stats')

    @classmethod
    def rollback_factor_stats(cls , rollback_date : int):
        """
        Rollback factor statistics tables via ``FactorCalculatorAPI.Stats``.

        [API Interaction]:
          expose: true
          email: true
          roles: [admin]
          risk: destructive
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        wrap_update(FactorCalculatorAPI.Stats.rollback , 'rollback factor stats' , rollback_date = rollback_date)

    @classmethod
    def test_factor_performance(
        cls , names : str = '' ,
        factor_type : Literal['factor' , 'pred'] = 'factor' , 
        benchmark : list[str] | str | Literal['defaults'] = 'defaults' , 
        start : int = 20240101 , end : int = 20240331 , step : int = 5 ,
        write_down : bool = True , display_figs : bool = False , indent : int = 0 , vb_level : Any = 1):
        """
        Run FactorPerf test on factors resolved via ``get_factor``.

        Args:
          names : Factor name(s) to test. If '' or 'random', a random factor will be used; Factor name(s) will be split by ','.
          factor_type : Factor type ('factor' or 'pred').
          benchmark : Benchmark name(s) to test. If 'defaults', will use all default benchmarks. If a string, will be split by ','.
          start : Start date for the test.
          end : End date for the test.
          step : Step size for the test.
          write_down : Whether to write down the test results.
          display_figs : Whether to display the test figures.
          indent : Indent level for the test.
          vb_level : Verbosity level for the test.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          override_args_attrs:
            benchmark :
              type: str
              default: defaults
          risk: read_only
          lock_num: 5
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        """
        with Logger.Paragraph('test factor performance' , 3):
            factor = get_factor(names , factor_type , start , end , step)
            if isinstance(benchmark , str) and ',' in benchmark:
                benchmark = benchmark.split(',')
            ret = FactorTestAPI.FactorPerf(factor , benchmark , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs)
        return ret
    
    @classmethod
    def test_optimized_fmp(
        cls , names : str = '' , factor_type : Literal['factor' , 'pred'] = 'factor' , 
        benchmark : list[str] | str | Literal['defaults'] = 'defaults' , 
        start : int = 20240101 , end : int = 20240331 , step : int = 5 ,
        write_down : bool = True , display_figs : bool = False , indent : int = 0 , vb_level : Any = 1 , 
        prob_type : Literal['linprog' , 'quadprog' , 'socp'] = 'linprog' ,
    ):
        """
        Run optimized FMP diagnostics via ``FactorTestAPI.FmpOptim``.

        Args:
          names : Factor name(s) to test. If '' or 'random', a random factor will be used; Factor name(s) will be split by ','.
          factor_type : Factor type ('factor' or 'pred').
          benchmark : Benchmark name(s) to test. If 'defaults', will use all default benchmarks. If a string, will be split by ','.
          start : Start date for the test.
          end : End date for the test.
          step : Step size for the test.
          write_down : Whether to write down the test results.
          display_figs : Whether to display the test figures.
          indent : Indent level for the test.
          vb_level : Verbosity level for the test.
          prob_type : Problem type for the optimization.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          override_args_attrs:
            benchmark :
              type: str
              default: defaults
          risk: read_only
          lock_num: 2
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        with Logger.Paragraph('test optimized fmp' , 3):
            factor = get_factor(names , factor_type , start , end , step)
            if isinstance(benchmark , str) and ',' in benchmark:
                benchmark = benchmark.split(',')
            ret = FactorTestAPI.FmpOptim(factor , benchmark , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs , prob_type = prob_type)
        return ret
    
    @classmethod
    def test_top_fmp(
        cls , names : str = '' , factor_type : Literal['factor' , 'pred'] = 'factor' , 
        benchmark : list[str] | str | Literal['defaults'] = 'defaults' , 
        start : int = 20240101 , end : int = 20240331 , step : int = 5 ,
        write_down : bool = True , display_figs : bool = False , indent : int = 0 , vb_level : Any = 1 , 
    ):
        """
        Run top fmp test on factors resolved via ``get_factor``.

        Args:
          names : Factor name(s) to test. If '' or 'random', a random factor will be used; Factor name(s) will be split by ','.
          factor_type : Factor type ('factor' or 'pred').
          benchmark : Benchmark name(s) to test. If 'defaults', will use all default benchmarks. If a string, will be split by ','.
          start : Start date for the test.
          end : End date for the test.
          step : Step size for the test.
          write_down : Whether to write down the test results.
          display_figs : Whether to display the test figures.
          indent : Indent level for the test.
          vb_level : Verbosity level for the test.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          override_args_attrs:
            benchmark :
              type: str
              default: defaults
          risk: read_only
          lock_num: 2
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        with Logger.Paragraph('test top fmp' , 3):
            factor = get_factor(names , factor_type , start , end , step)
            if isinstance(benchmark , str) and ',' in benchmark:
                benchmark = benchmark.split(',')
            ret = FactorTestAPI.FmpTop(factor , benchmark , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs)
        return ret

    @classmethod
    def fast_analyze(cls , factor_name : str , start : int = 20170101 , end : int = 99991231 , step : int = 10):
        """
        Fast analyze a factor over a given date range.

        Args:
          factor_name : Factor name to analyze.
          start : Start date for the analysis.
          end : End date for the analysis.
          step : Step size for the analysis.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: read_only
          lock_num: 5
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        """
        factor_calc = StockFactorHierarchy.get_factor(factor_name)
        dates = factor_calc.FactorDates(start,end,step)
        factor = factor_calc.Factor(dates , normalize = True , ignore_error = False)
        factor.fast_analyze(nday = step)
        return factor
    
    @classmethod
    def full_analyze(cls , factor_name : str , start : int = 20170101 , end : int = 99991231 , step : int = 1):
        """
        Full analyze a factor over a given date range.

        Args:
          factor_name : Factor name to analyze.
          start : Start date for the analysis.
          end : End date for the analysis.
          step : Step size for the analysis.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: read_only
          lock_num: 2
          disable_platforms: []
          execution_time: long
          memory_usage: large
        """
        factor_calc = StockFactorHierarchy.get_factor(factor_name)
        dates = factor_calc.FactorDates(start,end,step)
        factor = factor_calc.Factor(dates , normalize = True , ignore_error = False)
        factor.full_analyze(nday = step)
        return factor

    @classmethod
    def resume_testing_factors(cls):
        """
        Resume testing for every factor listed in ``Options.available_factors``.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: write
          lock_num: 1
          disable_platforms: []
          execution_time: long
          memory_usage: large
        """
        from src.api import ModelAPI
        for factor in Options.available_factors():
            title = f'Resume Testing Factor {factor}'
            with Logger.Paragraph(title , 1) if Proj.vb.vb > 1 else Logger.Timer(title):
                ModelAPI.test_factor(factor , resume = 1)