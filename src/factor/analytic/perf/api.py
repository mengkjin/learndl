import pandas as pd

from IPython.display import display 
from pathlib import Path
from typing import Any , Literal , Optional , Type

from src.data import DataBlock
from src.func import dfs_to_excel , figs_to_pdf
from src.factor.util import Benchmark

from . import calculator as Calc

class PerfManager:
    '''
    Factor Performance Calculator Manager
    Parameters:
        which : str | list[str] | Literal['all']
            Which tasks to run. Can be any of the following:
            'ic_curve' : IC Cumulative Curve
            'ic_decay' : IC Decay
            'ic_indus' : IC Industry
            'ic_year' : IC Year Stats
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
    CALC_DICT : dict[str,Type[Calc.BasePerfCalc]] = {
        'ic_curve' : Calc.IC_Cum_Curve , 
        'ic_decay' : Calc.IC_Decay ,
        'ic_indus' : Calc.IC_Industry ,
        'ic_year'  : Calc.IC_Year_Stats ,
        'ic_mono'  : Calc.IC_Monotony ,
        'pnl_curve' : Calc.PnL_Cum_Curve ,
        'style_corr' : Calc.Factor_Style_Corr ,
        'grp_curve' : Calc.Group_Ret_Cum_Curve ,
        # 'grp_decay_ret' : Calc.Group_Ret_Decay ,
        'grp_decay_ir' :  Calc.GroupDecayIR ,
        'grp_year' : Calc.GroupYearTop ,
        'distr_curve' : Calc.Distribution_Curve ,
        #'distr_qtile' : Calc.Distribution_Quantile ,
    }
    
    def __init__(self , which : str | list[str] | Literal['all'] = 'all' , **kwargs):
        if which == 'all':
            which = list(self.CALC_DICT.keys())
        elif isinstance(which , str):
            which = [which]
        assert all([t in self.CALC_DICT for t in which]) , f'Invalid task: {which}'
        self.tasks = {task:self.CALC_DICT[task]() for task in which}

    def calc(self , factor_val: DataBlock | pd.DataFrame, benchmarks: Optional[list[Benchmark|Any]] | Any = None , verbosity = 1):
        for name , task in self.tasks.items(): 
            task.calc(factor_val , benchmarks)
            if verbosity > 1: print(f'{self.__class__.__name__} calc of {name} Finished!')
        return self

    def plot(self , show = False , verbosity = 1):
        for name , task in self.tasks.items(): 
            task.plot(show = show)
            if verbosity > 1: print(f'{self.__class__.__name__} plot of {name} Finished!')
        if verbosity > 0: print(f'{self.__class__.__name__} plot Finished!')
        return self
    
    def save_rslts_and_figs(self , path : str):
        for name , task in self.tasks.items(): task.save(path = path)
        return self
    
    def get_rslts(self):
        return {name:task.calc_rslt for name , task in self.tasks.items()}
    
    def get_figs(self):
        return {f'{name}.{fig_name}':fig for name , task in self.tasks.items() for fig_name , fig in task.figs.items()}
    
    def display_figs(self):
        figs = self.get_figs()
        [display(fig) for fig in figs.values()]
        return figs
    
    def write_down(self , path : Path | str):
        path = Path(path)
        rslts = self.get_rslts()
        figs = self.get_figs()
        dfs_to_excel(rslts , path.joinpath('data.xlsx') , print_prefix='Analytic datas')
        figs_to_pdf(figs , path.joinpath('plot.pdf') , print_prefix='Analytic plots')

        return self
    
    @classmethod
    def run_test(cls , factor_val : pd.DataFrame | DataBlock , benchmark : list[str|Benchmark|Any] | Any = None ,
                 all = True , verbosity = 2 , **kwargs):
        pm = cls(all=all , **kwargs)
        pm.calc(factor_val , Benchmark.get_benchmarks(benchmark) , verbosity = verbosity).plot(show=False , verbosity = verbosity)
        return pm