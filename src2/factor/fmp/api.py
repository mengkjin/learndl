import pandas as pd

from dataclasses import asdict , dataclass , field
from IPython.display import display
from matplotlib.figure import Figure
from typing import Any , Optional

from . import calculator as Calc
from .builder import group_optimize
from .stat import group_accounting , calc_fmp_account
from ..loader import factor
from ..util import Benchmark , AlphaModel
from ...basic import PATH , DEFAULT_BENCHMARKS
from ...data import DataBlock
from ...func import dfs_to_excel , figs_to_pdf

@dataclass
class FmpManager:
    perf_params : dict[str,Any] = field(default_factory=dict)
    all : bool = False
    prefix : bool = True 
    perf_curve  : bool = False
    perf_drawdown  : bool = False
    perf_year  : bool = False
    perf_month   : bool = False
    perf_lag    : bool = False
    exp_style  : bool = False
    exp_indus   : bool = False
    attrib_source : bool = False
    attrib_style  : bool = False
    
    def __post_init__(self) -> None:
        self.perf_calc_dict = {k:self.select_perf_calc(k,self.perf_params) for k,v in self.perf_calc_boolean.items() if v}

    def optim(self , factor_val: DataBlock | pd.DataFrame, benchmarks: Optional[list[Benchmark|Any]] | Any = DEFAULT_BENCHMARKS , 
              add_lag = 1 , config_path = None , verbosity = 2):
        if isinstance(factor_val , DataBlock): factor_val = factor_val.to_dataframe()
        alpha_models = [AlphaModel.from_dataframe(factor_val[[factor_name]]) for factor_name in factor_val.columns]

        self.optim_tuples = group_optimize(alpha_models , benchmarks , add_lag , config_path = config_path , verbosity = verbosity)
        self.optim_tuples = group_accounting(self.optim_tuples , verbosity=verbosity)
        self.account = calc_fmp_account(self.optim_tuples)

    def calc(self , verbosity = 1):
        assert hasattr(self , 'account') , 'Must run FmpManager.optim first'
        for perf_name , perf_calc in self.perf_calc_dict.items(): 
            perf_calc.calc(self.account)
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
    
    def rslt_to_excel(self , path : str):
        assert path.endswith('.xlsx') , path
        rslts = self.get_rslts()
        dfs_to_excel(rslts , path , 'w' , 'fmp_')

    def figs_to_pdf(self , path : str):
        assert path.endswith('.pdf') , path
        figs = self.get_figs()
        figs_to_pdf(figs , path)

    @property
    def perf_calc_boolean(self) -> dict[str,bool]:
        return {k:bool(v) or self.all for k,v in asdict(self).items() if k not in ['perf_params' , 'all']}

    @staticmethod
    def select_perf_calc(key , param) -> Calc.BaseFmpCalc:
        return {
            'prefix' : Calc.Fmp_Prefix , 
            'perf_curve' : Calc.Fmp_Perf_Curve ,
            'perf_drawdown' : Calc.Fmp_Perf_Drawdown , 
            'perf_year'  : Calc.Fmp_Year_Stats ,
            'perf_month'  : Calc.Fmp_Month_Stats ,
            'perf_lag'    : Calc.Fmp_Perf_Lag_Curve ,
            'exp_style' : Calc.Fmp_Style_Exposure ,
            'exp_indus' : Calc.Fmp_Inustry_Deviation ,
            'attrib_source' : Calc.Fmp_Attribution_Source ,
            'attrib_style' : Calc.Fmp_Attribution_Style ,
        }[key](**param)
    
    @classmethod
    def run_test(cls , factor_val : pd.DataFrame | DataBlock , benchmark : list[Benchmark|Any] | Any | None = DEFAULT_BENCHMARKS ,
                 all = True , config_path : Optional[str] = None , verbosity = 2 , **kwargs):
        pm = cls(all=all , **kwargs)
        pm.optim(factor_val , benchmark , config_path=config_path , verbosity = verbosity)
        pm.calc(verbosity = verbosity).plot(show=False , verbosity = verbosity)
        return pm
    
    @classmethod
    def random_test(cls , nfactor = 1 , config_path :str | None = f'{PATH.conf}/fmp/custom.yaml' , verbosity = 2):
        factor_val = factor.random(20231201 , 20240228 , nfactor=nfactor)
        benchmark  = DEFAULT_BENCHMARKS
        pm = cls.run_test(factor_val , benchmark , all = True , config_path = config_path , verbosity = verbosity)
        return pm
