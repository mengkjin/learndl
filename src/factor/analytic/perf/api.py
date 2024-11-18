import pandas as pd

from dataclasses import asdict , dataclass , field
from IPython.display import display 
from matplotlib.figure import Figure
from pathlib import Path
from typing import Any , Optional

from . import calculator as Calc
from ...util import Benchmark
from ....data import DataBlock
from ....func import dfs_to_excel , figs_to_pdf

@dataclass
class PerfManager:
    perf_params : dict[str,Any] = field(default_factory=dict)
    all : bool = False
    ic_curve : bool = False
    ic_decay : bool = False
    ic_indus : bool = False
    ic_year  : bool = False
    ic_mono  : bool = False
    pnl_curve : bool = False
    style_corr : bool = False
    grp_curve : bool = False
    #grp_decay_ret : bool = False
    grp_decay_ir : bool = False
    grp_year : bool = False
    distr_curve : bool = False
    #distr_qtile : bool = False
    
    def __post_init__(self) -> None:
        self.perf_calc_dict = {k:self.select_perf_calc(k,self.perf_params) for k,v in self.perf_calc_boolean.items() if v}

    def calc(self , factor_val: DataBlock | pd.DataFrame, benchmarks: Optional[list[Benchmark|Any]] | Any = None , verbosity = 1):
        for perf_name , perf_calc in self.perf_calc_dict.items(): 
            perf_calc.calc(factor_val , benchmarks)
            if verbosity > 1: print(f'{self.__class__.__name__} calc of {perf_name} Finished!')
        if verbosity > 0: print(f'{self.__class__.__name__} calc Finished!')
        return self

    def plot(self , show = False , verbosity = 1):
        for perf_name , perf_calc in self.perf_calc_dict.items(): 
            perf_calc.plot(show = show)
            if verbosity > 1: print(f'{self.__class__.__name__} plot of {perf_name} Finished!')
        if verbosity > 0: print(f'{self.__class__.__name__} plot Finished!')
        return self
    
    def save_rslts_and_figs(self , path : str):
        for perf_name , perf_calc in self.perf_calc_dict.items(): perf_calc.save(path = path)
        return self
    
    def get_rslts(self):
        rslt : dict[str,pd.DataFrame] = {}
        for perf_key , perf_calc in self.perf_calc_dict.items():
            rslt[perf_key] = perf_calc.calc_rslt
        return rslt
    
    def get_figs(self):
        rslt : dict[str,Figure] = {}
        for perf_key , perf_calc in self.perf_calc_dict.items():
            [rslt.update({f'{perf_key}.{fig_name}':fig}) for fig_name , fig in perf_calc.figs.items()]
        return rslt
    
    def display_figs(self):
        figs = self.get_figs()
        [display(fig) for key , fig in figs.items()]
        return figs
    
    def write_down(self , path : Path | str):
        path = Path(path)
        rslts = self.get_rslts()
        figs = self.get_figs()
        dfs_to_excel(rslts , path.joinpath('data.xlsx') , print_prefix='Analytic datas')
        figs_to_pdf(figs , path.joinpath('plot.pdf') , print_prefix='Analytic plots')

        return self

    @property
    def perf_calc_boolean(self) -> dict[str,bool]:
        return {k:bool(v) or self.all for k,v in asdict(self).items() if k not in ['perf_params' , 'all']}

    @staticmethod
    def select_perf_calc(key , param) -> Calc.BasePerfCalc:
        return {
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
        }[key](**param)
    
    @classmethod
    def run_test(cls , factor_val : pd.DataFrame | DataBlock , benchmark : list[str|Benchmark|Any] | Any = None ,
                 all = True , verbosity = 2 , **kwargs):
        pm = cls(all=all , **kwargs)
        bms = Benchmark.get_benchmarks(benchmark)
        pm.calc(factor_val , bms , verbosity = verbosity).plot(show=False , verbosity = verbosity)
        return pm