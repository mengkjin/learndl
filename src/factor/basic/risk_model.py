import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any , ClassVar , Literal , Optional

from src.data import GetData, BlockLoader , FrameLoader , get_target_dates , load_target_file
from src.environ import RISK_INDUS , RISK_STYLE
from .port import Port

COMMON_FACTORS = ['market'] + RISK_INDUS + RISK_STYLE

@dataclass
class Rmodel:
    date : int
    F : pd.DataFrame
    C : pd.DataFrame
    S : pd.DataFrame

    def __post_init__(self):
        comfac = self.common_factors
        self.C = self.C.loc[comfac , comfac]
        self.S.fillna(self.S.quantile(0.95) , inplace=True)

    @property
    def secid(self): return self.F.index.values
    @property
    def universe(self): return self.F[self.F['estuniv'] == 1].index.values
    def ffmv(self , secid : np.ndarray | Any = None): 
        return self.weight(secid) * 1e8
    def weight(self , secid : np.ndarray | Any = None): 
        df = self.F.loc[: , 'weight'] * 1e8
        if secid is not None: df = df.loc[secid]
        return df.to_numpy().flatten()
    @property
    def common_factors(self): return COMMON_FACTORS
    def FCS_aligned(self , secid : np.ndarray | Any = None):
        F = self.F.loc[: , COMMON_FACTORS]
        C = self.C
        S = self.S
        if secid is not None: 
            F = F.loc[secid]
            S = S.loc[secid]
        return F.values.T , C.values , S.values.flatten()
    def industry(self , secid : np.ndarray | Any = None):
        df = self.F.loc[: , RISK_INDUS].idxmax(axis=1)
        if secid is not None: df = df.loc[secid]
        return df.to_numpy()
    def style(self , secid : np.ndarray | Any = None , style : np.ndarray | Any = None):
        if style is None: style = RISK_STYLE
        df = self.F.loc[: , style]
        if secid is not None: df = df.loc[secid]
        return df
    def is_pos_def(self):
        return np.all(np.linalg.eigvals(self.C.values) > 0)
    def _exposure(self , port : Port):
        rmodel_f = self.F.loc[port.secid , self.common_factors]
        return port.weight.dot(rmodel_f)
    def _specific_risk(self , port : Port):
        rmodel_s = self.S.loc[port.secid].values.flatten()
        return (port.weight * rmodel_s).dot(port.weight)
    def _analysis(self , port : Port | Any = None):
        if port is None: return RiskProfile()
        common_exp = self._exposure(port)
        variance = common_exp.dot(self.C).dot(common_exp.T) + self._specific_risk(port)
        return RiskProfile(self.common_factors , common_exp , variance)
    def analyze(self , port : Port , bench : Port | Any = None , init : Port | Any = None):
        rslt = Analytic()
        rslt.append('portfolio' , self._analysis(port))
        rslt.append('initial'   , self._analysis(init))
        rslt.append('benchmark' , self._analysis(bench))
        rslt.append('active'    , self._analysis(None if bench is None else port - bench))
        return rslt

class RiskModel:
    def __init__(self) -> None:
        self.models : dict[int,Rmodel] = {}
        self.riskmodel_available_dates = get_target_dates('models' , 'risk_exp')
        self.F_loader = BlockLoader('models' , 'risk_exp')
        self.C_loader = FrameLoader('models' , 'risk_cov')
        self.S_loader = BlockLoader('models' , 'risk_spec')

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.models)} days loaded)'

    def append(self , rmodel : Rmodel , override = False):
        assert override or (rmodel.date not in self.models.keys()) , rmodel.date
        self.models[rmodel.date] = rmodel

    def available_dates(self): return self.riskmodel_available_dates

    def latest_avail_date(self , date : int = 99991231):
        available_dates = self.available_dates()
        if date in available_dates: return date
        tar_dates = available_dates[available_dates < date]
        return max(tar_dates) if len(tar_dates) else -1
    
    def get(self , date : int , latest = True):
        use_date = self.latest_avail_date(date) if latest else date
        rmodel = self.models.get(use_date , None)

        if rmodel is None and use_date in self.available_dates():
            rmodel = self.load_day_model(date)
            self.append(rmodel)

        assert isinstance(rmodel , Rmodel) , f'rmodel does not exists!'
        return rmodel
    
    def load_models(self , dates : np.ndarray | Any = None , start : int = -1 , end : int = -1):
        if dates is None:
            dates = self.riskmodel_available_dates
            dates = [(dates >= start) & (dates <= end)]
        for date in np.setdiff1d(dates , list(self.models.keys())): self.models[date] = self.load_day_model(date) 

    def load_day_model(self , date : int):
        with GetData.Silence:
            F = self.F_loader.load(date , date)
            C = self.C_loader.load(date , date)
            S = self.S_loader.load(date , date)
        F = F.to_dataframe().reset_index(['date'],drop=True)
        C = C.reset_index(drop=True).set_index('factor_name')
        S = S.to_dataframe().reset_index(['date'],drop=True)
        return Rmodel(date , F , C , S)
    

@dataclass
class RiskProfile:
    factors  : Optional[np.ndarray | list] = None
    exposure : Optional[np.ndarray] = None
    variance : Optional[float] = None

    variance_measure : ClassVar[list[str]] = ['variance','std_dev']

    def __bool__(self): return self.factors is not None
    @property
    def standard_deviation(self): return np.sqrt(self.variance) if self.variance is not None else None
    def to_dataframe(self):
        if not self: return None
        df0 = pd.DataFrame({'factor_name':self.factors,'value':self.exposure})
        df1 = pd.DataFrame({'factor_name':self.variance_measure,'value':[self.variance,self.standard_deviation]})
        return pd.concat([df0,df1]).set_index('factor_name')
    
@dataclass
class Analytic:
    industry : pd.DataFrame | Any = None
    style    : pd.DataFrame | Any = None
    risk     : pd.DataFrame | Any = None

    def __post_init__(self):
        ...

    def __bool__(self): return self.industry is not None

    def append(self , port_type : Literal['portfolio' , 'benchmark' , 'initial' , 'active'] , risk_profile : RiskProfile):
        df = risk_profile.to_dataframe()
        if df is None: return
        industry = df.loc[RISK_INDUS].rename(columns={'value':port_type}).rename_axis('industry',axis='index')
        style    = df.loc[RISK_STYLE].rename(columns={'value':port_type}).rename_axis('style',axis='index')
        risk     = df.loc[RiskProfile.variance_measure].rename(columns={'value':port_type}).rename_axis('measure',axis='index')
        
        if self:
            self.industry = pd.concat([self.industry , industry] , axis = 1)
            self.style    = pd.concat([self.style    , style   ] , axis = 1)
            self.risk     = pd.concat([self.risk     , risk    ] , axis = 1)
        else:
            self.industry = industry
            self.style    = style
            self.risk     = risk

    def styler(self , which : Literal['industry' , 'style' , 'risk'] = 'style'):
        if which == 'industry':
            return self.industry.style.format(lambda x:f'{x:.2%}')
        elif which == 'style':
            return self.style.style.format(lambda x:f'{x:.4f}')
        else:
            return self.risk.style.format(lambda x:f'{x:.4%}')

RISK_MODEL = RiskModel()