from typing import Any , Type

from src.proj import Timer
from . import calculator as Calc
from ..test_manager import BaseTestManager
from ...util import Benchmark , StockFactor

__all__ = ['FactorPerfManager']

class FactorPerfManager(BaseTestManager):
    '''
    Factor Performance Calculator Manager
    Parameters:
        which : str | list[str] | Literal['all']
            Which tasks to run. Can be any of the following:
            'frontface' : Factor Front Face
            'coverage' : Factor Coverage
            'ic_curve' : IC Cumulative Curve
            'ic_decay' : IC Decay
            'ic_indus' : IC Industry
            'ic_year' : IC Year Stats
            'ic_benchmark' : IC Benchmark
            'ic_mono' : IC Monotony
            'pnl_curve' : PnL Cumulative Curve
            'style_corr' : Factor Style Correlation
            'grp_curve' : Group Return Cumulative Curve
            'grp_decay_ir' : Group Return Decay
            'grp_year' : Group Return Yearly Top
            'distr_curve' : Distribution Curve
        deprecated : 
            'grp_decay_ret' : Group Return Decay
            'distr_qtile' : Distribution Quantile
    '''
    TASK_TYPE = 'factor'
    TASK_LIST : list[Type[Calc.BasePerfCalc]] = [
        Calc.Factor_FrontFace ,
        Calc.Factor_Coverage ,
        Calc.Factor_IC_Curve , 
        Calc.Factor_IC_Decay ,
        Calc.Factor_IC_Indus ,
        Calc.Factor_IC_Year ,
        Calc.Factor_IC_Benchmark ,
        Calc.Factor_IC_Monotony ,
        Calc.Factor_PnL_Curve ,
        Calc.Factor_Style_Corr ,
        # Calc.Factor_Style_Corr_Distrib ,
        Calc.Factor_Group_Curve ,
        Calc.Factor_Group_Decay ,
        Calc.Factor_Group_IR_Decay ,
        Calc.Factor_Group_Year ,
        Calc.Factor_Distrib_Curve ,
        # Calc.Factor_Distrib_Qtile ,
    ]

    def calc(self , factor: StockFactor , benchmarks: list[Benchmark|Any] | Any = None , verbosity = 1):
        with Timer(f'{self.__class__.__name__} calc' , silent = verbosity < 1):
            for task in self.tasks.values(): 
                task.calc(factor , benchmarks , verbosity = verbosity - 1)
        return self