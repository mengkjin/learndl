from __future__ import annotations

from typing import Any , Literal , Iterable

from src.proj import Base
from src.res.factor.util import StockFactor
from src.res.factor.risk import TuShareCNE5_Calculator
from src.res.factor.analytic import (
    FactorPerfTest , OptimFMPTest , BaseFactorAnalyticTest , TopFMPTest)
from src.res.factor.calculator import (
    StockFactorHierarchy , StockFactorUpdater , MarketFactorUpdater , 
    AffiliateFactorUpdater , PoolingFactorUpdater , FactorStatsUpdater
)
from src.res.factor.calculator.updater import BaseFactorUpdater

class RiskModelUpdater:
    @classmethod
    def update(cls) -> Base.UpdateFlagList:
        flags = Base.UpdateFlagList()
        flags += TuShareCNE5_Calculator.update()
        return flags

    @classmethod
    def rollback(cls , rollback_date : int) -> Base.UpdateFlagList:
        flags = Base.UpdateFlagList()
        flags += TuShareCNE5_Calculator.rollback(rollback_date)
        return flags

class FactorUpdaterAPI:
    Stock = StockFactorUpdater()
    Market = MarketFactorUpdater()
    Affiliate = AffiliateFactorUpdater()
    Pooling = PoolingFactorUpdater()
    Stats = FactorStatsUpdater()

    @classmethod
    def updaters(cls) -> list[BaseFactorUpdater]:
        return [cls.Market , cls.Stock , cls.Affiliate , cls.Pooling , cls.Stats]

    @classmethod
    def update(cls , **kwargs) -> Base.UpdateFlagList:
        flags = Base.UpdateFlagList()
        for updater in cls.updaters():
            flags += updater.update(**kwargs)
        flags += cls.export()
        return flags

    @classmethod
    def rollback(cls , rollback_date : int , **kwargs) -> Base.UpdateFlagList:
        flags = Base.UpdateFlagList()
        for updater in cls.updaters():
            flags += updater.rollback(rollback_date , **kwargs)
        flags += cls.export()   
        return flags

    @classmethod
    def recalculate(cls , **kwargs) -> Base.UpdateFlagList:
        flags = Base.UpdateFlagList()
        for updater in cls.updaters():
            flags += updater.recalculate(**kwargs)
        flags += cls.export()
        return flags

    @classmethod
    def fix(cls , factors : list[str] , **kwargs) -> Base.UpdateFlagList:
        flags = Base.UpdateFlagList()
        for updater in cls.updaters():
            flags += updater.fix(factors , **kwargs)
        flags += cls.export()
        return flags

    @classmethod
    def export(cls):
        return StockFactorHierarchy.export_factor_table()

class FactorTestAPI:
    Hierarchy = StockFactorHierarchy

    @classmethod
    def get_test_class(cls , test_type : Base.TestType):
        """
        get the test class for the given test type , test type
        """
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
    def run_test(
        cls , test_type : Base.TestType | str , 
        factor : StockFactor , benchmarks : Base.alias.MultipleBenchmark = 'defaults' ,
        test_path : Base.strPath | None = None , 
        resume : bool = False , save_resumable : bool = False , 
        indent : int = 0 , vb_level : Any = 1 , start : int = -1 , end : int = 99991231 ,
        write_down = False , display_figs = False , **kwargs
    ):
        test_type = Base.TestType(test_type)
        testor = cls.create(
            test_type , test_path , resume , save_resumable , start , end , 
            indent = indent , vb_level = vb_level , **kwargs
        )
        testor.proceed(factor , benchmarks)
        if write_down:   
            testor.write_down()
        if display_figs: 
            testor.display_figs()
        return testor

    @classmethod
    def create(cls , test_type : Base.TestType ,
               test_path : Base.strPath | None = None , resume : bool = False , save_resumable : bool = False ,
               start : int = -1 , end : int = 99991231 ,**kwargs):
        testor_type = cls.get_test_class(test_type)
        testor = testor_type.create(test_path , resume , save_resumable , start , end , **kwargs)
        return testor

    @classmethod
    def last_portfolio_date(cls , test_path : Base.strPath , test_types : Base.TestType | Iterable[Base.TestType] | Literal['all']):
        last_portfolio_dates = []
        for test_type in Base.TestType.ensure_list(test_types):
            last_portfolio_date = cls.get_test_class(test_type).last_portfolio_date(test_path)
            last_portfolio_dates.append(last_portfolio_date)
        return min(last_portfolio_dates) if len(last_portfolio_dates) else 19000101

    @classmethod
    def factor_stats_saved_dates(cls , test_path : Base.strPath):
        return cls.get_test_class(Base.TestType.FACTOR).factor_stats_saved_dates(test_path)

    @classmethod
    def FactorPerf(
        cls , factor : StockFactor , benchmarks : Base.alias.MultipleBenchmark = 'defaults' ,
        test_path : Base.strPath | None = None , resume : bool = False , 
        indent : int = 0 , vb_level : Any = 1 , start : int = -1 , end : int = 99991231 ,
        write_down = False , display_figs = False , save_resumable : bool = False , **kwargs
    ):
        pm = cls.run_test('factor' , factor , benchmarks , test_path , resume , save_resumable , 
                          indent , vb_level , start , end , write_down , display_figs , **kwargs)
        assert isinstance(pm , FactorPerfTest) , 'FactorPerfTest is expected!'
        return pm
    
    @classmethod
    def FmpOptim(
        cls , factor : StockFactor , benchmarks : Base.alias.MultipleBenchmark = 'defaults' , 
        test_path : Base.strPath | None = None , resume : bool = False , 
        indent : int = 0 , vb_level : Any = 1 , start : int = -1 , end : int = 99991231 ,
        write_down = False , display_figs = False , save_resumable : bool = False , **kwargs
    ):
        pm = cls.run_test('optim' , factor , benchmarks , test_path , resume , save_resumable , 
                          indent , vb_level , start , end , write_down , display_figs , **kwargs)
        assert isinstance(pm , OptimFMPTest) , 'OptimFMPTest is expected!'
        return pm


    @classmethod
    def FmpTop(
        cls , factor : StockFactor , benchmarks : Base.alias.MultipleBenchmark = 'defaults' , 
        test_path : Base.strPath | None = None , resume : bool = False , 
        indent : int = 0 , vb_level : Any = 1 , start : int = -1 , end : int = 99991231 ,
        write_down = False , display_figs = False , save_resumable : bool = False , **kwargs
    ):
        pm = cls.run_test('top' , factor , benchmarks , test_path , resume , save_resumable , 
                          indent , vb_level , start , end , write_down , display_figs , **kwargs)
        assert isinstance(pm , TopFMPTest) , 'TopFMPTest is expected!'
        return pm