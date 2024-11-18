import pandas as pd

from IPython.display import display
from pathlib import Path
from typing import Any , Literal , Optional , Type

from src.func import dfs_to_excel , figs_to_pdf
from src.data import DataBlock

from .builder import PortfolioBuilderGroup
from . import calculator as Calc
from ...util import AlphaModel , Benchmark

class FmpManager:
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
    CALC_DICT : dict[str,Type[Calc.BaseFmpCalc]] = {
        'prefix'        : Calc.Fmp_Prefix , 
        'perf_curve'    : Calc.Fmp_Perf_Curve ,
        'perf_drawdown' : Calc.Fmp_Perf_Drawdown , 
        'perf_year'     : Calc.Fmp_Year_Stats ,
        'perf_month'    : Calc.Fmp_Month_Stats ,
        'perf_lag'      : Calc.Fmp_Perf_Lag_Curve ,
        'exp_style'     : Calc.Fmp_Style_Exposure ,
        'exp_indus'     : Calc.Fmp_Inustry_Deviation ,
        'attrib_source' : Calc.Fmp_Attribution_Source ,
        'attrib_style'  : Calc.Fmp_Attribution_Style ,
    }

    def __init__(self , which : str | list[str] | Literal['all'] = 'all' , **kwargs):
        if which == 'all':
            which = list(self.CALC_DICT.keys())
        elif isinstance(which , str):
            which = [which]
        assert all([t in self.CALC_DICT for t in which]) , f'Invalid task: {which}'
        self.tasks = {task:self.CALC_DICT[task]() for task in which}

    def optim(self , factor_val: DataBlock | pd.DataFrame, benchmarks: Optional[list[Benchmark|Any]] | Any = 'defaults' , 
              add_lag = 1 , config_path = None , verbosity = 2):
        if isinstance(factor_val , DataBlock): factor_val = factor_val.to_dataframe()
        alpha_models = [AlphaModel.from_dataframe(factor_val[[factor_name]]) for factor_name in factor_val.columns]
        bms = Benchmark.get_benchmarks(benchmarks)
        self.portfolio_group = PortfolioBuilderGroup('optim' , alpha_models , bms , add_lag , optim_config_path = config_path , verbosity = verbosity)
        self.account = self.portfolio_group.building().accounting().total_account()

    def calc(self , verbosity = 1):
        assert hasattr(self , 'account') , 'Must run FmpManager.optim first'
        for name , task in self.tasks.items(): 
            task.calc(self.account)
            if verbosity > 1: print(f'{self.__class__.__name__} calc of {name} Finished!')
        if verbosity > 0: print(f'{self.__class__.__name__} calc Finished!')
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
        dfs_to_excel(rslts , path.joinpath('data.xlsx') , print_prefix='Model Portfolio datas')
        figs_to_pdf(figs , path.joinpath('plot.pdf') , print_prefix='Model Portfolio plots')
        return self
    
    @classmethod
    def run_test(cls , factor_val : pd.DataFrame | DataBlock , benchmark : list[Benchmark|Any] | Any | None = 'defaults' ,
                 all = True , config_path : Optional[str] = None , verbosity = 2 ,**kwargs):
        pm = cls(all=all , **kwargs)
        pm.optim(factor_val , Benchmark.get_benchmarks(benchmark) , config_path=config_path , verbosity = verbosity)
        pm.calc(verbosity = verbosity).plot(show=False , verbosity = verbosity)
        return pm
