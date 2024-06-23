import pandas as pd

from dataclasses import asdict , dataclass , field
from IPython.display import display 
from typing import Any , Literal , Optional

from src.data import DataBlock
from ..loader import factor
from ..basic import Benchmark
from . import util as U

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

    def calc(self , factor_val: DataBlock | pd.DataFrame, benchmarks: Optional[list[Benchmark|Any]] | Any = None):
        for _ , perf_calc in self.perf_calc_dict.items(): perf_calc.calc(factor_val , benchmarks)
        return self

    def plot(self , show = False):
        for _ , perf_calc in self.perf_calc_dict.items(): perf_calc.plot(show = show)
        return self
    
    def save(self , path : str):
        for perf_name , perf_calc in self.perf_calc_dict.items(): perf_calc.save(path = path)
        return self
    
    def get_rslts(self):
        rslt = {}
        for perf_key , perf_calc in self.perf_calc_dict.items():
            rslt[perf_key] = perf_calc.calc_rslt
        return rslt
    
    def get_figs(self):
        rslt = {}
        for perf_key , perf_calc in self.perf_calc_dict.items():
            [rslt.update({f'{perf_key}.{fig_name}':fig}) for fig_name , fig in perf_calc.figs.items()]
        return rslt

    @property
    def perf_calc_boolean(self) -> dict[str,bool]:
        return {k:bool(v) or self.all for k,v in asdict(self).items() if k not in ['perf_params' , 'all']}

    @staticmethod
    def select_perf_calc(key , param) -> U.BasePerfCalc:
        return {
            'ic_curve' : U.IC_Cum_Curve , 
            'ic_decay' : U.IC_Decay ,
            'ic_indus' : U.IC_Industry ,
            'ic_year'  : U.IC_Year_Stats ,
            'ic_mono'  : U.IC_Monotony ,
            'pnl_curve' : U.PnL_Cum_Curve ,
            'style_corr' : U.Factor_Style_Corr ,
            'grp_curve' : U.Group_Ret_Cum_Curve ,
            # 'grp_decay_ret' : U.Group_Ret_Decay ,
            'grp_decay_ir' :  U.GroupDecayIR ,
            'grp_year' : U.GroupYearTop ,
            'distr_curve' : U.Distribution_Curve ,
            #'distr_qtile' : U.Distribution_Quantile ,
        }[key](**param)
    
    @classmethod
    def run_test(cls , factor_val : pd.DataFrame | DataBlock , benchmark : list[Benchmark|Any] | Any = None ,
                 all = True , **kwargs):
        pm = cls(all=all , **kwargs)
        pm.calc(factor_val , benchmark).plot(show=False)
        return pm
    
    @classmethod
    def random_test(cls , nfactor = 1):
        factor_val = factor.random(20231201 , 20240228 , nfactor=nfactor)
        benchmark  = None # Benchmark('csi500')
        return cls.run_test(factor_val , benchmark)
