"""
Factor API for the project
"""
from __future__ import annotations
from typing import Any

from src.proj import Logger , Proj , Options , Base
from src.proj.util.filesys.ttl_cache import DiskTTLCache
from src.data import DATAVENDOR

from src.res.factor.util import StockFactor
from src.res.factor.api import RiskModelUpdater , FactorUpdaterAPI , FactorTestAPI
from src.res.factor.calculator import StockFactorHierarchy

from src.api.util import wrap_update

__all__ = [
    'FactorAPI' , 'get_random_factor' , 'get_real_factor' , 'get_factor' , 
    'RiskModelUpdater' , 'FactorUpdaterAPI' , 'FactorTestAPI' , 'StockFactorHierarchy' ,
    'StockFactor' , 'DATAVENDOR'
]

def get_random_factor(start = 20240101 , end = 20240331 , step = 5 , default_random_n = 2):
    """
    Build a ``StockFactor`` from a random synthetic vendor slice (debug/experiments).
    """
    return StockFactor(DATAVENDOR.random_factor(start , end , step , default_random_n).to_dataframe())

def get_real_factor(names : str | list[str] | None = None ,
                    factor_type : Base.lit.FactorType = 'factor' , 
                    start = 20240101 , end = 20240331 , step = 5):
    """
    Load named real factors/preds from ``DATAVENDOR`` into a ``StockFactor``.
    """
    assert names and names != 'random' , 'Names are required and not random for real factor!'
    return StockFactor(DATAVENDOR.real_factor(factor_type , names , start , end , step).to_dataframe())

def get_factor(names : str | list[str] | None = None , 
               factor_type : Base.lit.FactorType = 'factor' , 
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
        record_entry = DiskTTLCache.get('daily_update', 'export_factor_table')
        if record_entry.valid_value:
            def bypass_export():
                Logger.skipping(f'Factor table already exported {record_entry.time_str} ...' , indent = 1)
                return Base.UpdateFlag.SKIPPED
            flag = wrap_update(bypass_export , 'export factor table')
        else:
            flag = wrap_update(StockFactorHierarchy.export_factor_table , 'export factor table')
            if flag == Base.UpdateFlag.SUCCESS:
                record_entry.put(True , ttl_hours = 24)
        return flag

    @classmethod
    def update_stock_factors(cls , timeout : int = -1):
        """
        Update stock factors via ``FactorUpdaterAPI.Stock``.

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
        wrap_update(FactorUpdaterAPI.Stock.update , 'update stock factors' , timeout = timeout)

    @classmethod
    def rollback_stock_factors(cls , rollback_date : int , timeout : int = -1):
        """
        Rollback stock factors via ``FactorUpdaterAPI.Stock``.

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
        wrap_update(FactorUpdaterAPI.Stock.rollback , 'rollback stock factors' , rollback_date = rollback_date , timeout = timeout)
    
    @classmethod
    def update_market_factors(cls):
        """
        Update market-wide factors via ``FactorUpdaterAPI.Market``.

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
        wrap_update(FactorUpdaterAPI.Market.update , 'update market factors')

    @classmethod
    def rollback_market_factors(cls , rollback_date : int):
        """
        Rollback market-wide factors via ``FactorUpdaterAPI.Market``.

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
        wrap_update(FactorUpdaterAPI.Market.rollback , 'rollback market factors' , rollback_date = rollback_date)

    @classmethod
    def update_affiliate_factors(cls):
        """
        Update affiliate / external risk factors via ``FactorUpdaterAPI.Affiliate``.

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
        wrap_update(FactorUpdaterAPI.Affiliate.update , 'update risk & external factors')

    @classmethod
    def rollback_affiliate_factors(cls , rollback_date : int):
        """
        Rollback affiliate / external risk factors via ``FactorUpdaterAPI.Affiliate``.

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
        wrap_update(FactorUpdaterAPI.Affiliate.rollback , 'rollback risk & external factors' , rollback_date = rollback_date)

    @classmethod
    def update_pooling_factors(cls , timeout : int = -1):
        """
        Update pooling-layer factors via ``FactorUpdaterAPI.Pooling``.

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
        wrap_update(FactorUpdaterAPI.Pooling.update , 'update pooling factors' , timeout = timeout)

    @classmethod
    def rollback_pooling_factors(cls , rollback_date : int , timeout : int = -1):
        """
        Rollback pooling-layer factors via ``FactorUpdaterAPI.Pooling``.

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
        wrap_update(FactorUpdaterAPI.Pooling.rollback , 'rollback pooling factors' , rollback_date = rollback_date , timeout = timeout)

    @classmethod
    def update_factor_stats(cls):
        """
        Update factor statistics tables via ``FactorUpdaterAPI.Stats``.

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
        wrap_update(FactorUpdaterAPI.Stats.update , 'update factor stats')

    @classmethod
    def rollback_factor_stats(cls , rollback_date : int):
        """
        Rollback factor statistics tables via ``FactorUpdaterAPI.Stats``.

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
        wrap_update(FactorUpdaterAPI.Stats.rollback , 'rollback factor stats' , rollback_date = rollback_date)

    @classmethod
    def test_factor_performance(
        cls , names : str = '' ,
        factor_type : Base.lit.FactorType = 'factor' , 
        benchmarks : Base.alias.MultipleBenchmark = 'defaults' , 
        start : int = 20240101 , end : int = 20240331 , step : int = 5 ,
        write_down : bool = True , display_figs : bool = False , indent : int = 0 , vb_level : Any = 1):
        """
        Run FactorPerf test on factors resolved via ``get_factor``.

        Args:
          names : Factor name(s) to test. If '' or 'random', a random factor will be used; Factor name(s) will be split by ','.
          factor_type : Factor type ('factor' or 'pred').
          benchmarks : Benchmark name(s) to test. If 'defaults', will use all default benchmarks. If a string, will be split by ','.
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
          override_arg_attr:
            benchmarks :
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
            if isinstance(benchmarks , str) and ',' in benchmarks:
                benchmarks = benchmarks.split(',')
            ret = FactorTestAPI.FactorPerf(factor , benchmarks , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs)
        return ret
    
    @classmethod
    def test_optimized_fmp(
        cls , names : str = '' , factor_type : Base.lit.FactorType = 'factor' , 
        benchmarks : Base.alias.MultipleBenchmark = 'defaults' , 
        start : int = 20240101 , end : int = 20240331 , step : int = 5 ,
        write_down : bool = True , display_figs : bool = False , indent : int = 0 , vb_level : Any = 1 , 
        prob_type : Base.PortOptimProblem | str = Base.PortOptimProblem.LINPROG ,
    ):
        """
        Run optimized FMP diagnostics via ``FactorTestAPI.FmpOptim``.

        Args:
          names : Factor name(s) to test. If '' or 'random', a random factor will be used; Factor name(s) will be split by ','.
          factor_type : Factor type ('factor' or 'pred').
          benchmarks : Benchmark name(s) to test. If 'defaults', will use all default benchmarks. If a string, will be split by ','.
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
          override_arg_attr:
            benchmarks :
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
            if isinstance(benchmarks , str) and ',' in benchmarks:
                benchmarks = benchmarks.split(',')
            ret = FactorTestAPI.FmpOptim(factor , benchmarks , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs , prob_type = prob_type)
        return ret
    
    @classmethod
    def test_top_fmp(
        cls , names : str = '' , factor_type : Base.lit.FactorType = 'factor' , 
        benchmarks : Base.alias.MultipleBenchmark = 'defaults' , 
        start : int = 20240101 , end : int = 20240331 , step : int = 5 ,
        write_down : bool = True , display_figs : bool = False , indent : int = 0 , vb_level : Any = 1 , 
    ):
        """
        Run top fmp test on factors resolved via ``get_factor``.

        Args:
          names : Factor name(s) to test. If '' or 'random', a random factor will be used; Factor name(s) will be split by ','.
          factor_type : Factor type ('factor' or 'pred').
          benchmarks : Benchmark name(s) to test. If 'defaults', will use all default benchmarks. If a string, will be split by ','.
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
          override_arg_attr:
            benchmarks :
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
            if isinstance(benchmarks , str) and ',' in benchmarks:
                benchmarks = benchmarks.split(',')
            ret = FactorTestAPI.FmpTop(factor , benchmarks , indent = indent , vb_level = vb_level , write_down = write_down , display_figs = display_figs)
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
          memory_usage: high
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
          memory_usage: high
        """
        from src.api.pkgs import ModelAPI
        for factor in Options.available_factors():
            title = f'Resume Testing Factor {factor}'
            with Logger.Paragraph(title , 1) if Proj.vb.vb > 1 else Logger.Timer(title):
                ModelAPI.test_factor(factor , resume = 1)

    @classmethod
    def recalculate_all_factors(cls , start : int | None = None , end : int | None = None , timeout : int = 10):
        """
        Recalculate factors via ``FactorUpdaterAPI``.

        Args:
          start : Start date for the recalculation.
          end : End date for the recalculation.
          timeout : Timeout for the recalculation.

        Returns:
          None

        [API Interaction]:
          expose: true
          email: true
          roles: [admin]
          risk: destructive
          lock_num: 1
          disable_platforms: [macos , linux , windows]
          execution_time: long
          memory_usage: high
        """
        wrap_update(FactorUpdaterAPI.recalculate , 'recalculate all factors' , start = start , end = end , timeout = timeout)