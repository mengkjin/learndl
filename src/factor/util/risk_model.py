import numpy as np
import pandas as pd
import statsmodels.api as sm

from dataclasses import dataclass
from typing import Any , ClassVar , Literal , Optional

from .classes import GeneralModel , Port
from ...basic import SILENT
from ...basic.conf import RISK_INDUS , RISK_STYLE , RISK_COMMON , ROUNDING_CONTRIBUTION , ROUNDING_EXPOSURE
from ...data import BlockLoader , FrameLoader , DATAVENDOR

@dataclass
class Rmodel:
    '''
    risk model instance for one day
    '''
    date : int
    F : pd.DataFrame
    C : pd.DataFrame
    S : pd.DataFrame

    def __post_init__(self):
        comfac = self.common_factors
        self.C = self.C.loc[comfac , comfac]
        self.S.fillna(self.S.quantile(0.95) , inplace=True)
        self.regressed : int | Any = None
        self.next_date = DATAVENDOR.td(self.date , 1).td

    def assert_valid(self):
        assert self.F.shape[-1] == self.C.shape[0] == self.S.shape[0] == self.C.shape[1], (self.F.shape , self.C.shape , self.S.shape)
        assert self.S.shape[1] == 1 , self.S.shape

    @property
    def secid(self): return self.F.index.values
    @property
    def universe(self): return self.F[self.F['estuniv'] == 1].index.values
    def ffmv(self , secid : np.ndarray | Any = None): 
        return self.weight(secid) * 1e8
    def weight(self , secid : np.ndarray | Any = None): 
        df = self.F.loc[: , 'weight'] * 1e8
        if secid is not None: df = self.loc_secid(df , secid , 0.)
        return df.to_numpy().flatten()
    @property
    def common_factors(self): return RISK_COMMON
    def FCS_aligned(self , secid : np.ndarray | Any = None):
        F = self.F.loc[: , RISK_COMMON]
        C = self.C
        S = self.S
        if secid is not None: 
            F = self.loc_secid(F , secid , 0.)
            S = self.loc_secid(S , secid , 'max')
        return F.values.T , C.values , S.values.flatten()
    def industry(self , secid : np.ndarray | Any = None):
        df = self.F.loc[: , RISK_INDUS].idxmax(axis=1)
        if secid is not None: df = self.loc_secid(df , secid , 0.)
        return df.to_numpy()
    def style(self , secid : np.ndarray | Any = None , style : np.ndarray | Any = None):
        if style is None: style = RISK_STYLE
        df = self.F.loc[: , style]
        if secid is not None: df = self.loc_secid(df , secid , 0.)
        return df
    def is_pos_def(self):
        return np.all(np.linalg.eigvals(self.C.values) > 0)
    def _exposure(self , port : Port):
        rmodel_f = self.loc_secid(self.F.loc[: , self.common_factors] , port.secid)
        return port.weight.dot(rmodel_f)
    def _specific_risk(self , port : Port):
        rmodel_s = self.loc_secid(self.S , port.secid).values.flatten()
        return (port.weight * rmodel_s).dot(port.weight)
    def _analysis(self , port : Port | Any = None):
        if not port: return RiskProfile()
        common_exp = self._exposure(port)
        variance = common_exp.dot(self.C).dot(common_exp.T) + self._specific_risk(port)
        return RiskProfile(self.common_factors , common_exp , variance)
    def analyze(self , port : Port , bench : Port | Any = None , init : Port | Any = None):
        '''Analyze day end risk profile'''
        rslt = RiskAnalytic(self.date)
        rslt.append('portfolio' , self._analysis(port))
        rslt.append('initial'   , self._analysis(init))
        rslt.append('benchmark' , self._analysis(bench))
        rslt.append('active'    , self._analysis(None if bench is None else port - bench))
        return rslt
    def attribute(self , port : Port , bench : Port | Any = None , target_date : Optional[int] = None , other_cost : float = 0.):
        '''Attribute the portfolio of the next day (trading day)'''
        rslt = Attribution.create(self , port , bench , target_date=target_date , other_cost = other_cost)
        return rslt
    def regress_fut_ret(self , target_date : Optional[int] = None):
        '''regress future day return , most likely daily , but can given any target date'''
        if target_date is None: target_date = self.next_date
        futret = DATAVENDOR.get_quote_ret(self.date , target_date)
        if futret is None: 
            self.params : pd.DataFrame | Any = None
            self.futret : pd.DataFrame | Any = None
            self.regressed = None
        else:
            futret = futret[['ret']].rename(columns={'ret':'tot'}).astype(float)
            F = self.F.join(futret).fillna(0)

            model = sm.WLS(F['tot'] , F['market'].fillna(0), weights=F['weight']).fit()   # type: ignore
            market_ret , excess_ret = model.params , model.resid.rename('excess')

            model = sm.WLS(excess_ret , F[RISK_INDUS + RISK_STYLE], weights=F['weight']).fit()   # type: ignore
            self.params = pd.concat([market_ret , model.params])
            self.futret = futret.loc[:,['tot']].join(excess_ret).join(model.resid.rename('specific'))
            self.regressed = target_date
        return self
    
    @staticmethod
    def loc_secid(df : pd.DataFrame | Any , secid : np.ndarray , fillna : Literal['max','min'] | float | None = None):    
        try:  
            new_df = df.loc[secid]
        except KeyError as e:  
            new_df = df.reindex(secid).loc[secid]
        if fillna is not None:
            if fillna == 'max':
                new_df = new_df.fillna(new_df.max())
            elif fillna == 'min':
                new_df = new_df.fillna(new_df.min())
            else:
                new_df = new_df.fillna(new_df)
        return new_df

class RiskModel(GeneralModel):
    '''
    risk model instance for multiple days
    '''
    _instance_dict : dict = {}
    _singleton_names = ['cne5']

    def __new__(cls , name : str | Any = 'cne5' , *args , **kwargs):
        if name in cls._instance_dict:
            return cls._instance_dict[name]
        elif name in cls._singleton_names:
            instance = super().__new__(cls , *args , **kwargs)
            cls._instance_dict[name] = instance
            return instance
        else:
            raise ValueError(f'{cls.__name__} does not have {name} as a riskmodel name!')
    
    def __init__(self , model_name = 'cne5') -> None:
        self.name = model_name
        if not getattr(self , 'models' , None):
            self.models : dict[int,Rmodel] = {}
            self.init_loaders()
        
    def init_loaders(self):
        self.riskmodel_available_dates = DATAVENDOR.file_dates('models' , 'tushare_cne5_exp')
        self.F_loader = BlockLoader('models' , 'tushare_cne5_exp')
        self.C_loader = FrameLoader('models' , 'tushare_cne5_cov')
        self.S_loader = BlockLoader('models' , 'tushare_cne5_spec')

    def append(self , model : Rmodel , override = False):
        return super().append(model , override)
    def available_dates(self): return self.riskmodel_available_dates
    def get(self , date : int , latest = True) -> Rmodel:
        model = super().get(date , latest)
        assert isinstance(model , Rmodel) , f'rmodel does not exists!'
        return model
    def load_day_model(self , date : int):
        with SILENT:
            F = self.F_loader.load(date , date)
            C = self.C_loader.load(date , date)
            S = self.S_loader.load(date , date)
        F = F.to_dataframe().assign(market=1.).reset_index(['date'],drop=True).astype(float)
        C = C.drop(columns='date').set_index('factor_name').astype(float)
        S = S.to_dataframe().reset_index(['date'],drop=True).astype(float)
        return Rmodel(date , F , C , S)    

@dataclass
class RiskProfile:
    '''
    basic risk profile of a portfolio / benchmark / active , i.e. risk exposure / standard deviation / variance
    '''
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
class RiskAnalytic:
    '''
    portfolio risk analysis result
    '''
    date     : int = 0
    industry : pd.DataFrame | Any = None
    style    : pd.DataFrame | Any = None
    risk     : pd.DataFrame | Any = None

    def __post_init__(self):
        ...
    def __bool__(self): return bool(self.date > 0)
    def __repr__(self):
        return f'{self.__class__.__name__}({self.date})'

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

    def rounding(self):
        if self.industry is not None: self.industry = self.industry.round(ROUNDING_EXPOSURE)
        if self.style    is not None: self.style    = self.style.round(ROUNDING_EXPOSURE)
        if self.risk     is not None: self.risk     = self.risk.round(ROUNDING_EXPOSURE)
        return self

    def styler(self , which : Literal['industry' , 'style' , 'risk'] = 'style'):
        if which == 'industry':
            return self.industry.style.format(lambda x:f'{x:.2%}')
        elif which == 'style':
            return self.style.style.format(lambda x:f'{x:.4f}')
        else:
            return self.risk.style.format(lambda x:f'{x:.4%}')

@dataclass
class Attribution:
    '''
    portfolio risk attribution result
    '''
    start    : int = 0
    end      : int = 0
    source   : pd.DataFrame | Any = None
    industry : pd.DataFrame | Any = None
    style    : pd.DataFrame | Any = None
    specific : pd.DataFrame | Any = None
    aggregated : pd.DataFrame | Any = None

    def __post_init__(self):
        ...
    def __bool__(self): return self.start > 0 and self.end > 0
    def __repr__(self):
        return f'{self.__class__.__name__}({self.start}-{self.end})'

    @classmethod
    def create(cls , risk_model : Rmodel , port : Port , bench : Optional[Port] = None , 
               target_date : Optional[int] = None , other_cost : float = 0.):
        if target_date is None: target_date = risk_model.next_date
        if risk_model.regressed != target_date: risk_model = risk_model.regress_fut_ret(target_date)
        if risk_model.futret is None: return cls(risk_model.next_date , risk_model.regressed)

        benchport = bench.port if bench else Port.EMPTY_PORT
        weight = port.port.merge(benchport , on='secid' , how='outer').set_index('secid').fillna(0)
        weight.columns = ['portfolio' , 'benchmark']
        weight['active'] = weight['portfolio'] - weight['benchmark']

        futret = risk_model.futret
        F = risk_model.F.drop(columns=['estuniv','weight'])
        coef = risk_model.params

        specific = weight.join(futret).join(F).drop(columns = weight.columns)
        aggregated = []
        for col in weight.columns:
            exp = (specific * weight[col].to_numpy().reshape(-1,1)).sum().rename(col).to_frame()
            exp.loc[futret.columns.values] = weight[col].sum()
            aggregated.append(exp)

        specific *= weight['active'].to_numpy().reshape(-1,1)
        specific.loc[:,coef.index.values] *= coef.to_numpy().reshape(1,-1)
        aggregated.append(specific.sum().rename('contribution'))
        aggregated = pd.concat(aggregated , axis = 1)
    
        agg_cost = pd.DataFrame([[0,0,0,-other_cost]], columns = aggregated.columns,index=['cost'])
        aggregated = pd.concat([aggregated , agg_cost]).rename_axis('source' , axis = 'index')
        aggregated.loc[['tot'],['contribution']] = aggregated.loc[['tot'],['contribution']] - other_cost
        
        industry = aggregated.loc[RISK_INDUS]
        style    = aggregated.loc[RISK_STYLE]

        source   = pd.concat([aggregated.loc[['tot','market','excess','specific','cost']] ,
                              industry.sum().rename('industry').to_frame().T ,
                              style.sum().rename('style').to_frame().T])
        source.loc[['industry','style'] , weight.columns] = 0.

        order_list = ['tot','market','industry','style','excess','specific','cost']
        source = source.loc[order_list].rename_axis('source' , axis = 'index')

        specific['industry'] = specific.loc[:,RISK_INDUS].sum(1)
        specific['style']    = specific.loc[:,RISK_STYLE].sum(1)
        specific = specific.drop(columns = RISK_INDUS + RISK_STYLE).loc[:,order_list[:-1]]

        return cls(risk_model.next_date , risk_model.regressed , source , industry , style , specific , aggregated)
    
    def rounding(self):
        decimals = {
            'portfolio' : ROUNDING_EXPOSURE , 
            'benchmark' : ROUNDING_EXPOSURE , 
            'active'    : ROUNDING_EXPOSURE , 
            'contribution' : ROUNDING_CONTRIBUTION}
        if self.source is not None:     self.source   = self.source.round(decimals)
        if self.industry is not None:   self.industry = self.industry.round(decimals)
        if self.style    is not None:   self.style    = self.style.round(decimals)
        if self.aggregated is not None: self.aggregated = self.aggregated.round(decimals)
        return self
    
RISK_MODEL = RiskModel()