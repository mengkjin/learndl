import pandas as pd

from dataclasses import asdict , dataclass , field
from typing import Any , Literal , Optional

from src.data import DataBlock
from ..loader import factor
from ..basic import Benchmark , BENCHMARKS , AlphaModel , AVAIL_BENCHMARK

from .stat import group_accounting , calc_fmp_account
from .builder import group_optimize
from . import util as U


@dataclass
class FmpManager:
    perf_params : dict[str,Any] = field(default_factory=dict)
    all : bool = False
    prefix : bool = True 
    perf_decay  : bool = False
    perf_year  : bool = False
    perf_month   : bool = False
    perf_lag    : bool = False
    exp_style  : bool = False
    exp_indus  : bool = False
    attrib_curve   : bool = False
    attrib_style   : bool = False
    
    def __post_init__(self) -> None:
        self.perf_calc_dict = {k:self.select_perf_calc(k,self.perf_params) for k,v in self.perf_calc_boolean.items() if v}

    def optim(self , factor_val: DataBlock | pd.DataFrame, benchmarks: Optional[list[Benchmark|Any]] | Any = AVAIL_BENCHMARK , 
              lags = [0,1,2] , config_path = None , verbosity = 2):
        if isinstance(factor_val , DataBlock): factor_val = factor_val.to_dataframe()
        alpha_models = [AlphaModel.from_dataframe(factor_val[[factor_name]]) for factor_name in factor_val.columns]

        self.optim_tuples = group_optimize(alpha_models , benchmarks , lags , config_path = config_path , verbosity = verbosity)
        self.optim_tuples = group_accounting(self.optim_tuples , verbosity=verbosity)
        self.account = calc_fmp_account(self.optim_tuples)

    def calc(self):
        assert hasattr(self , 'account') , 'Must run FmpManager.optim first'
        for _ , perf_calc in self.perf_calc_dict.items(): perf_calc.calc(self.account)
        return self

    def plot(self , show = False):
        for _ , perf_calc in self.perf_calc_dict.items(): perf_calc.plot(show = show)
        return self
    
    def save(self , path : str):
        for perf_name , perf_calc in self.perf_calc_dict.items(): perf_calc.save(path = path , key = perf_name)
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
    def select_perf_calc(key , param) -> U.BaseFmpCalc:
        return {
            'prefix' : U.FmpPrefix , 
            'perf_decay' : U.FmpPerfCurve ,
            'perf_year' : U.FmpPerfYear ,
            'perf_month'  : U.FmpPerfMonth ,
            'perf_lag'    : U.FmpLagCurve ,
            'exp_style' : U.FmpStyleExp ,
            'exp_indus' : U.FmpInustryExp ,
            'attrib_curve' : U.FmpAttributionCurve ,
            'attrib_style' : U.FmpAttributionStyleCurve ,
        }[key](**param)
    
    @classmethod
    def random_test(cls , nfactor = 1 , config_path  = 'custom_opt_config.yaml' , verbosity = 2):
        factor_val = factor.random(20231201 , 20240228 , nfactor=nfactor)
        # benchmark  = None # Benchmark('csi500')

        pm = cls(all=True)
        pm.optim(factor_val , config_path=config_path , verbosity = verbosity)
        pm.calc().plot(show=False)
        return pm
