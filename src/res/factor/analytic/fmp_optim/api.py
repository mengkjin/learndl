from typing import Any , Literal , Type

from src.proj import Timer
from . import calculator as Calc
from ..test_manager import BaseTestManager
from ...util import Benchmark , StockFactor
from ...fmp import PortfolioBuilderGroup

__all__ = ['OptimFMPTest']

class OptimFMPTest(BaseTestManager):
    '''
    Factor Model PortfolioPerformance Calculator Manager
    Parameters:
        which : str | list[str] | Literal['all']
            Which tasks to run. Can be any of the following:
            'prefix' : Prefix
            'perf_curve' : Performance Curve
            'perf_drawdown' : Performance Drawdown
            'perf_year' : Performance Yearly Stats
            'perf_month' : Performance Monthly Stats
            'perf_lag' : Performance Lag Curve
            'exp_style' : Style Exposure
            'exp_indus' : Industry Deviation
            'attrib_source' : Attribution Source
            'attrib_style' : Attribution Style
    '''
    TASK_TYPE = 'optim'
    TASK_LIST : list[Type[Calc.OptimCalc]] = [
        Calc.FrontFace , 
        Calc.Perf_Curve ,
        Calc.Drawdown , 
        Calc.Excess_Drawdown , 
        Calc.Perf_Year ,
        Calc.Perf_Month ,
        Calc.Perf_Lag ,
        Calc.Exp_Style ,
        Calc.Exp_Indus ,
        Calc.Attrib_Source ,
        Calc.Attrib_Style ,
    ]

    def optim(self , factor: StockFactor , benchmarks: list[Benchmark|Any] | Any = 'defaults' , 
              add_lag = 1 , optim_config = None , verbosity = 2):
        with Timer(f'{self.__class__.__name__}.get_alpha_models'):
            alpha_models = factor.alpha_models()
        with Timer(f'{self.__class__.__name__}.get_benchmarks'):
            benchmarks = Benchmark.get_benchmarks(benchmarks)
        self.update_kwargs(add_lag = add_lag , optim_config = optim_config , verbosity = verbosity)
        self.portfolio_group = PortfolioBuilderGroup('optim' , alpha_models , benchmarks , resume = self.resume , resume_path = self.resume_path , **self.kwargs)
        self.account = self.portfolio_group.building().accounting().total_account()

    def calc(self , factor : StockFactor , benchmark : list[Benchmark|Any] | Any | None = 'defaults' ,
             add_lag = 1 , optim_config : str | Literal['default' , 'custome'] | None = None , verbosity = 1 , **kwargs):
        self.optim(factor , benchmark , add_lag = add_lag ,optim_config = optim_config , verbosity = verbosity)
        with Timer(f'{self.__class__.__name__}.calc' , silent = verbosity < 1):
            for task in self.tasks.values():  
                task.calc(self.account , verbosity = verbosity - 1) 
        return self
    
    def update_kwargs(self , add_lag = 1 , **kwargs):
        self.kwargs.update({
            'add_lag': 1 if any([task in self.tasks for task in ['perf_lag']]) else add_lag ,
            'analytic':any([task in self.tasks for task in ['exp_style' , 'exp_indus']]) ,
            'attribution':any([task in self.tasks for task in ['attrib_source' , 'attrib_style']])})
        self.kwargs.update(kwargs)
        return self
