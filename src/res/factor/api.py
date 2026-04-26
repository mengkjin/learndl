from typing import Any , Literal

from src.proj.core import strPath
from src.res.factor.util import StockFactor
from src.res.factor.risk import TuShareCNE5_Calculator
from src.res.factor.analytic import (
    TEST_TYPES , TYPE_of_TEST , FactorPerfTest , OptimFMPTest , BaseFactorAnalyticTest , TopFMPTest)
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
    def get_test_class(cls , test_type : TYPE_of_TEST):
        """get the test class for the given test type , test type"""
        return BaseFactorAnalyticTest.get_test_class(test_type)

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
                 test_path : strPath | None = None , 
                 resume : bool = False , save_resumable : bool = False , 
                 indent : int = 0 , vb_level : Any = 1 , start : int = -1 , end : int = 99991231 ,
                 write_down = False , display_figs = False , **kwargs):
        testor = cls.create(test_type , test_path , resume , save_resumable , start , end , **kwargs)
        testor.proceed(factor , benchmark , indent = indent , vb_level = vb_level)
        if write_down:   
            testor.write_down()
        if display_figs: 
            testor.display_figs()
        return testor

    @classmethod
    def create(cls , test_type : TYPE_of_TEST ,
               test_path : strPath | None = None , resume : bool = False , save_resumable : bool = False ,
               start : int = -1 , end : int = 99991231 ,**kwargs):
        testor_type = cls.get_test_class(test_type)
        testor = testor_type.create(test_path , resume , save_resumable , start , end , **kwargs)
        return testor

    @classmethod
    def last_portfolio_date(cls , test_path : strPath , test_types : list[TYPE_of_TEST] | TYPE_of_TEST):
        if not isinstance(test_types , list):
            test_types = [test_types]
        last_portfolio_dates = []
        for test_type in test_types:
            last_portfolio_date = cls.get_test_class(test_type).last_portfolio_date(test_path)
            last_portfolio_dates.append(last_portfolio_date)
        return min(last_portfolio_dates) if len(last_portfolio_dates) else 19000101

    @classmethod
    def factor_stats_saved_dates(cls , test_path : strPath):
        return cls.get_test_class('factor').factor_stats_saved_dates(test_path)

    @classmethod
    def FactorPerf(cls , factor : StockFactor , benchmark : list[str] | str | Literal['defaults'] = 'defaults' ,
                   test_path : strPath | None = None , resume : bool = False , 
                   indent : int = 0 , vb_level : Any = 1 , start : int = -1 , end : int = 99991231 ,
                   write_down = False , display_figs = False , save_resumable : bool = False , **kwargs):
        pm = cls.run_test('factor' , factor , benchmark , test_path , resume , save_resumable , 
                          indent , vb_level , start , end , write_down , display_figs , **kwargs)
        assert isinstance(pm , FactorPerfTest) , 'FactorPerfTest is expected!'
        return pm
    
    @classmethod
    def FmpOptim(cls , factor : StockFactor , benchmark : list[str] | str | Literal['defaults'] = 'defaults' , 
                 test_path : strPath | None = None , resume : bool = False , 
                 indent : int = 0 , vb_level : Any = 1 , start : int = -1 , end : int = 99991231 ,
                 write_down = False , display_figs = False , save_resumable : bool = False , **kwargs):
        pm = cls.run_test('optim' , factor , benchmark , test_path , resume , save_resumable , 
                          indent , vb_level , start , end , write_down , display_figs , **kwargs)
        assert isinstance(pm , OptimFMPTest) , 'OptimFMPTest is expected!'
        return pm


    @classmethod
    def FmpTop(cls , factor : StockFactor , benchmark : list[str] | str | Literal['defaults'] = 'defaults' , 
               test_path : strPath | None = None , resume : bool = False , 
               indent : int = 0 , vb_level : Any = 1 , start : int = -1 , end : int = 99991231 ,
               write_down = False , display_figs = False , save_resumable : bool = False , **kwargs):
        pm = cls.run_test('top' , factor , benchmark , test_path , resume , save_resumable , 
                          indent , vb_level , start , end , write_down , display_figs , **kwargs)
        assert isinstance(pm , TopFMPTest) , 'TopFMPTest is expected!'
        return pm