from typing import Any , Literal
from pathlib import Path

from src.res.factor.util import StockFactor
from src.res.factor.risk import TuShareCNE5_Calculator
from src.res.factor.analytic import TEST_TYPES , TYPE_of_TEST , FactorPerfTest , OptimFMPTest , TopFMPTest , T50FMPTest , ScreenFMPTest , RevScreenFMPTest
from src.res.factor.calculator import (
    StockFactorHierarchy , StockFactorUpdater , MarketFactorUpdater , 
    AffiliateFactorUpdater , PoolingFactorUpdater , FactorStatsUpdater
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
    Affiliate = AffiliateFactorUpdater
    Pooling = PoolingFactorUpdater
    Stats = FactorStatsUpdater

    @classmethod
    def update(cls , **kwargs):
        cls.Market.update(**kwargs)
        cls.Stock.update(**kwargs)
        cls.Affiliate.update(**kwargs)
        cls.Pooling.update(**kwargs)
        cls.Stats.update(**kwargs)
        cls.export()

    @classmethod
    def rollback(cls , rollback_date : int , **kwargs):
        cls.Market.rollback(rollback_date , **kwargs)
        cls.Stock.rollback(rollback_date , **kwargs)
        cls.Affiliate.rollback(rollback_date , **kwargs)
        cls.Pooling.rollback(rollback_date , **kwargs)
        cls.Stats.rollback(rollback_date , **kwargs)
        cls.export()

    @classmethod
    def recalculate(cls , **kwargs):
        cls.Market.recalculate(**kwargs)
        cls.Stock.recalculate(**kwargs)
        cls.Affiliate.recalculate(**kwargs)
        cls.Pooling.recalculate(**kwargs)
        cls.Stats.recalculate(**kwargs)
        cls.export()

    @classmethod
    def fix(cls , factors : list[str] , **kwargs):
        cls.Market.fix(factors , **kwargs)
        cls.Stock.fix(factors , **kwargs)
        cls.Affiliate.fix(factors , **kwargs)
        cls.Pooling.fix(factors , **kwargs)
        cls.Stats.fix(factors , **kwargs)
        cls.export()

    @classmethod
    def export(cls):
        StockFactorHierarchy.export_factor_table()

class FactorTestAPI:
    TEST_TYPES = TEST_TYPES
    Hierarchy = StockFactorHierarchy

    @classmethod
    def get_analytic_test(cls , test_type : TYPE_of_TEST):
        if test_type == 'factor':
            return FactorPerfTest
        elif test_type == 'optim':
            return OptimFMPTest
        elif test_type == 'top':
            return TopFMPTest
        elif test_type == 't50':
            return T50FMPTest
        elif test_type == 'screen':
            return ScreenFMPTest
        elif test_type == 'revscreen':
            return RevScreenFMPTest
        else:
            raise ValueError(f'Invalid test type: {test_type}')

    @classmethod
    def get_test_name(cls , test_type : TYPE_of_TEST):
        return cls.get_analytic_test(test_type).__name__

    @classmethod
    def FastAnalyze(cls , factor_name : str , start : int | None = 20170101 , end : int | None = None , step : int = 10 , lag = 2):
        calc = cls.Hierarchy.get_factor(factor_name)
        factor = calc.Factor(calc.FactorDates(start,end,step))
        factor.fast_analyze()
        return factor

    @classmethod
    def FullAnalyze(cls , factor_name : str , start : int | None = 20170101 , end : int | None = None , step : int = 1 , lag = 2):
        calc = cls.Hierarchy.get_factor(factor_name)
        factor = calc.Factor(calc.FactorDates(start,end,step))
        factor.full_analyze()
        return factor
        
    @classmethod
    def run_test(cls , test_type : TYPE_of_TEST , 
                 factor : StockFactor , benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' ,
                 test_name : str | None = None , test_path : Path | str | None = None , 
                 resume : bool = False , save_resumable : bool = False , 
                 indent : int = 0 , vb_level : int = 1 , start_dt : int = -1 , end_dt : int = 99991231 ,
                 write_down = False , display_figs = False , **kwargs):
        pm = cls.get_analytic_test(test_type).run_test(
            factor , benchmark , test_name , test_path , resume , save_resumable , indent , vb_level , start_dt , end_dt , **kwargs)
        if write_down:   
            pm.write_down()
        if display_figs: 
            pm.display_figs()
        return pm

    @classmethod
    def FactorPerf(cls , factor : StockFactor , benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' ,
                   test_name : str | None = None , test_path : Path | str | None = None , resume : bool = False , 
                   indent : int = 0 , vb_level : int = 1 , start_dt : int = -1 , end_dt : int = 99991231 ,
                   write_down = False , display_figs = False , save_resumable : bool = False , **kwargs):
        pm = cls.run_test('factor' , factor , benchmark , test_name , test_path , resume , save_resumable , 
                          indent , vb_level , start_dt , end_dt , write_down , display_figs , **kwargs)
        assert isinstance(pm , FactorPerfTest) , 'FactorPerfTest is expected!'
        return pm
    
    @classmethod
    def FmpOptim(cls , factor : StockFactor , benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                 test_name : str | None = None , test_path : Path | str | None = None , resume : bool = False , 
                 indent : int = 0 , vb_level : int = 1 , start_dt : int = -1 , end_dt : int = 99991231 ,
                 write_down = False , display_figs = False , save_resumable : bool = False , **kwargs):
        pm = cls.run_test('optim' , factor , benchmark , test_name , test_path , resume , save_resumable , 
                          indent , vb_level , start_dt , end_dt , write_down , display_figs , **kwargs)
        assert isinstance(pm , OptimFMPTest) , 'OptimFMPTest is expected!'
        return pm


    @classmethod
    def FmpTop(cls , factor : StockFactor , benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
               test_name : str | None = None , test_path : Path | str | None = None , resume : bool = False , 
               indent : int = 0 , vb_level : int = 1 , start_dt : int = -1 , end_dt : int = 99991231 ,
               write_down = False , display_figs = False , save_resumable : bool = False , **kwargs):
        pm = cls.run_test('top' , factor , benchmark , test_name , test_path , resume , save_resumable , 
                          indent , vb_level , start_dt , end_dt , write_down , display_figs , **kwargs)
        assert isinstance(pm , TopFMPTest) , 'TopFMPTest is expected!'
        return pm