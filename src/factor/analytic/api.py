import pandas as pd
from typing import Any , Literal

from src.data import DataBlock

from .test_manager import TASK_TYPES , TYPE_of_TASK
from .factor_perf.api import FactorPerfManager
from .fmp_optim.api import FmpOptimManager
from .fmp_top.api import FmpTopManager

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
        else:
            raise ValueError(f'Invalid test type: {test_type}')
        
    @staticmethod
    def _print_test_info(test_type : TYPE_of_TASK , factor_val : DataBlock | pd.DataFrame ,
                 benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' ,
                 write_down = False , display_figs = False):
        n_factor = factor_val.shape[-1]
        if isinstance(benchmark , list):
            benchmark_str = f'{len(benchmark)} BMs'
        else:
            benchmark_str = f'BM [{str(benchmark)}]'
        print(f'Running {test_type} test for {n_factor} factors in {benchmark_str}, write_down={write_down}, display_figs={display_figs}')

    @classmethod
    def run_test(cls , test_type : Literal['factor' , 'optim' , 'top'] , 
                 factor_val : DataBlock | pd.DataFrame ,
                 benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' ,
                 write_down = False , display_figs = False , verbosity = 1 , **kwargs):
        if verbosity > 0: cls._print_test_info(test_type , factor_val , benchmark , write_down , display_figs)
        test_manager = cls.get_test_manager(test_type)
        pm = test_manager.run_test(factor_val , benchmark , verbosity=verbosity , **kwargs)
        if write_down:   pm.write_down()
        if display_figs: pm.display_figs()
        return pm

    @classmethod
    def FactorPerf(cls , factor_val : DataBlock | pd.DataFrame ,
                   benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' ,
                   write_down = False , display_figs = False , verbosity = 1 , **kwargs):
        pm = cls.run_test('factor' , factor_val , benchmark , write_down , display_figs , verbosity , **kwargs)
        assert isinstance(pm , FactorPerfManager) , 'FactorPerfManager is expected!'
        return pm
    
    @classmethod
    def FmpOptim(cls , factor_val : DataBlock | pd.DataFrame ,
                 benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                 write_down = False , display_figs = False , verbosity = 1 , **kwargs):
        pm = cls.run_test('optim' , factor_val , benchmark , write_down , display_figs , verbosity , **kwargs)
        assert isinstance(pm , FmpOptimManager) , 'FmpOptimManager is expected!'
        return pm


    @classmethod
    def FmpTop(cls , factor_val : DataBlock | pd.DataFrame ,
                benchmark : list[str|Any] | str | Any | Literal['defaults'] = 'defaults' , 
                write_down = False , display_figs = False , verbosity = 1 , **kwargs):
        pm = cls.run_test('top' , factor_val , benchmark , write_down , display_figs , verbosity , **kwargs)
        assert isinstance(pm , FmpTopManager) , 'FmpTopManager is expected!'
        return pm