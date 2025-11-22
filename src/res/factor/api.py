from typing import Any , Literal

from src.res.factor.util import StockFactor
from src.res.factor.risk import TuShareCNE5_Calculator
from src.res.factor.analytic import TASK_TYPES , TYPE_of_TASK , FactorPerfManager , FmpOptimManager , FmpTopManager , FmpT50Manager , FmpScreenManager
from src.res.factor.calculator import (
    StockFactorHierarchy , StockFactorUpdater , MarketFactorUpdater , 
    RiskFactorUpdater , PoolingFactorUpdater , FactorStatsUpdater
)

class RiskModelUpdater:
    @classmethod
    def update(cls):
        TuShareCNE5_Calculator.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        TuShareCNE5_Calculator.rollback(rollback_date)

class FactorCalculatorAPI:
    Stock = StockFactorUpdater
    Market = MarketFactorUpdater
    Risk = RiskFactorUpdater
    Pooling = PoolingFactorUpdater
    Stats = FactorStatsUpdater

    @classmethod
    def update(cls , **kwargs):
        cls.Stock.update(**kwargs)
        cls.Market.update(**kwargs)
        cls.Risk.update(**kwargs)
        cls.Pooling.update(**kwargs)
        cls.Stats.update(**kwargs)
        cls.export()

    @classmethod
    def rollback(cls , rollback_date : int , **kwargs):
        cls.Stock.rollback(rollback_date , **kwargs)
        cls.Market.rollback(rollback_date , **kwargs)
        cls.Risk.rollback(rollback_date , **kwargs)
        cls.Pooling.rollback(rollback_date , **kwargs)
        cls.Stats.rollback(rollback_date , **kwargs)
        cls.export()

    @classmethod
    def recalculate(cls , **kwargs):
        cls.Stock.recalculate(**kwargs)
        cls.Market.recalculate(**kwargs)
        cls.Risk.recalculate(**kwargs)
        cls.Pooling.recalculate(**kwargs)
        cls.Stats.recalculate(**kwargs)
        cls.export()

    @classmethod
    def fix(cls , factors : list[str] , **kwargs):
        cls.Stock.fix(factors , **kwargs)
        cls.Market.fix(factors , **kwargs)
        cls.Risk.fix(factors , **kwargs)
        cls.Pooling.fix(factors , **kwargs)
        cls.Stats.fix(factors , **kwargs)
        cls.export()

    @classmethod
    def export(cls):
        StockFactorHierarchy.export_factor_table()

class FactorTestAPI:
    TASK_TYPES = TASK_TYPES
    @classmethod
    def get_test_manager(cls , test_type : TYPE_of_TASK):
        if test_type == 'factor':
            return FactorPerfManager
        elif test_type == 'optim':
            return FmpOptimManager
        elif test_type == 'top':
            return FmpTopManager
        elif test_type == 't50':
            return FmpT50Manager
        elif test_type == 'screen':
            return FmpScreenManager
        else:
            raise ValueError(f'Invalid test type: {test_type}')
        
    @staticmethod
    def _print_test_info(test_type : TYPE_of_TASK , factor : StockFactor ,
                         benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' ,
                         write_down = False , display_figs = False):
        n_factor = len(factor.factor_names)
        benchmark_str = f'{len(benchmark)} BMs' if isinstance(benchmark , list) else f'BM [{str(benchmark)}]'
        print(f'Running {test_type} test for {n_factor} factors in {benchmark_str}, write_down={write_down}, display_figs={display_figs}')

    @classmethod
    def run_test(cls , test_type : TYPE_of_TASK , 
                 factor : StockFactor , benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' ,
                 write_down = False , display_figs = False , verbosity = 1 , 
                 project_name : str | None = None , **kwargs):
        if verbosity > 0: 
            cls._print_test_info(test_type , factor , benchmark , write_down , display_figs)
        test_manager = cls.get_test_manager(test_type)
        pm = test_manager.run_test(factor , benchmark , verbosity=verbosity , project_name = project_name , **kwargs)
        if write_down:   
            pm.write_down()
        if display_figs: 
            pm.display_figs()
        return pm

    @classmethod
    def FactorPerf(cls , factor : StockFactor , benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' ,
                   write_down = False , display_figs = False , verbosity = 1 , project_name : str | None = None , **kwargs):
        pm = cls.run_test('factor' , factor , benchmark , write_down , display_figs , verbosity , project_name = project_name , **kwargs)
        assert isinstance(pm , FactorPerfManager) , 'FactorPerfManager is expected!'
        return pm
    
    @classmethod
    def FmpOptim(cls , factor : StockFactor , benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                 write_down = False , display_figs = False , verbosity = 1 , project_name : str | None = None , **kwargs):
        pm = cls.run_test('optim' , factor , benchmark , write_down , display_figs , verbosity , project_name = project_name , **kwargs)
        assert isinstance(pm , FmpOptimManager) , 'FmpOptimManager is expected!'
        return pm


    @classmethod
    def FmpTop(cls , factor : StockFactor , benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
               write_down = False , display_figs = False , verbosity = 1 , project_name : str | None = None , **kwargs):
        pm = cls.run_test('top' , factor , benchmark , write_down , display_figs , verbosity , project_name = project_name , **kwargs)
        assert isinstance(pm , FmpTopManager) , 'FmpTopManager is expected!'
        return pm