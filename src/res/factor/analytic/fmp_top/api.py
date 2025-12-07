from typing import Any , Type

from src.proj import Timer
from . import calculator as Calc
from ..test_manager import BaseTestManager
from ...util import Benchmark , StockFactor
from ...fmp import PortfolioBuilderGroup

__all__ = ['FmpTopManager']

class FmpTopManager(BaseTestManager):
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
            'exp_style' : Style Exposure
            'exp_indus' : Industry Deviation
            'attrib_source' : Attribution Source
            'attrib_style' : Attribution Style
    '''
    TASK_TYPE = 'top'
    TASK_LIST : list[Type[Calc.BaseTopPortCalc]] = [
        Calc.FrontFace , 
        Calc.Perf_Curve ,
        Calc.Perf_Excess ,
        Calc.Drawdown , 
        Calc.Excess_Drawdown ,
        Calc.Perf_Year ,
        Calc.Exp_Style ,
        Calc.Exp_Indus ,
        Calc.Attrib_Source ,
        Calc.Attrib_Style ,
    ]

    def generate(self , factor: StockFactor , benchmarks: list[Benchmark|Any] | Any = 'defaults' , 
                 n_bests = [20,30,50,100] , verbosity = 2):
        alpha_models = factor.alpha_models()
        print(f'finish alpha_models')
        benchmarks = Benchmark.get_benchmarks(benchmarks)
        self.update_kwargs(n_bests = n_bests , verbosity = verbosity)
        self.portfolio_group = PortfolioBuilderGroup('top' , alpha_models , benchmarks , **self.kwargs)
        print(f'finish portfolio_group')
        self.account = self.portfolio_group.building().accounting().total_account()

    def calc(self , factor : StockFactor , benchmark : list[Benchmark|Any] | Any | None = 'defaults' ,
             n_bests = [20,30,50,100] , verbosity = 1 , **kwargs):
        with Timer(f'{self.__class__.__name__} calc' , silent = verbosity < 1):
            self.generate(factor , benchmark , n_bests = n_bests , verbosity = verbosity)
            for task in self.tasks.values():  
                task.calc(self.account , verbosity = verbosity - 1) 
        return self
    
    def update_kwargs(self , n_bests = [20,30,50,100] , **kwargs):
        self.kwargs.update({
            'add_lag': 0 ,# 1 if any([task in self.tasks for task in ['perf_lag']]) else 0 ,
            'analytic':any([task in self.tasks for task in ['exp_style' , 'exp_indus']]) ,
            'attribution':any([task in self.tasks for task in ['attrib_source' , 'attrib_style']])})
        self.kwargs['param_groups'] = {f'Top{n:3d}':{'n_best':n} for n in n_bests}
        self.kwargs.update(kwargs)
        return self