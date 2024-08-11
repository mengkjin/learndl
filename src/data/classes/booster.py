import torch
import numpy as np
import pandas as pd
import xarray as xr

from typing import Any , ClassVar , Literal , Optional

from .nd import NdData
from ...func import match_values

class BoosterData:
    SECID_COLS : ClassVar[list[str]] = ['SecID','instrument','secid']
    DATE_COLS  : ClassVar[list[str]] = ['TradeDate','datetime','date']

    def __init__(
        self , 
        x : pd.DataFrame | np.ndarray | torch.Tensor | NdData ,
        y : pd.Series    | np.ndarray | torch.Tensor | NdData ,
        secid   : Any = None ,
        date    : Any = None ,
        feature : Any = None ,
        weight_param : Optional[dict] = None
    ):
        assert len(x) == len(y) , f'x and y length must match'
        self.input_type = type(y)
        if isinstance(x , pd.DataFrame) and isinstance(y , pd.Series): 
            self.df_index = y.index
            self.var_sec  = [v for v in self.SECID_COLS if v in x.index.names][0]
            self.var_date = [v for v in self.DATE_COLS  if v in x.index.names][0]
            x = x.reset_index().set_index([self.var_sec,self.var_date])
            xarr = xr.Dataset.from_dataframe(x)
 
            xindex = [arr.values for arr in xarr.indexes.values()] + [list(xarr.data_vars)]
            x = np.stack([arr.to_numpy() for arr in xarr.data_vars.values()] , -1)

            yarr = xr.Dataset.from_dataframe(pd.DataFrame(y.reset_index().set_index([self.var_sec,self.var_date])))
            y = np.stack([arr.to_numpy() for arr in yarr.data_vars.values()] , -1)
        elif isinstance(x , NdData) and isinstance(y , NdData):
            xindex = x.index
            x = x.values
            y = y.values
        else:
            xindex = [None , None , None]

        if isinstance(x , torch.Tensor): x = x.detach().cpu().numpy()
        if isinstance(y , torch.Tensor): y = y.detach().cpu().numpy()

        assert isinstance(x , np.ndarray) and isinstance(y , np.ndarray) , (x,y)

        assert x.ndim in [2,3] , x.ndim
        assert y.ndim in [x.ndim - 1, x.ndim] , (y.ndim , x.ndim)
        
        if y.ndim == x.ndim:
            assert y.shape[-1] == 1 , f'Booster Data cannot deal with multilabels, but got {y.shape}'
            y = y[...,0]

        if x.ndim == 2: x , y = x[:,None,:] , y[:,None]

        if secid is None:  secid = xindex[0] if xindex[0] is not None else np.arange(x.shape[0])
        if date  is None : date  = xindex[1] if xindex[1] is not None else np.arange(x.shape[1])
        if feature is None : feature = xindex[-1] if xindex[-1] is not None else np.array([f'F.{i}' for i in range(x.shape[-1])])

        assert x.shape == (len(secid) , len(date) , len(feature)) , (x.shape , (len(secid) , len(date) , len(feature)))
        assert y.shape == (len(secid) , len(date)) , (y.shape , (len(secid) , len(date)))
        
        self.x , self.y , self.finite = x , y , np.isfinite(y)

        self.update_feature()
        if weight_param is None: 
            weight_param = {'tau':0.75*np.log(0.5)/np.log(0.75) , 'ts_type':'lin' , 'rate':0.5}  

        self.secid , self.date , self.feature , self.weight_param = secid , date , feature , weight_param

    def __repr__(self):
        return '\n'.join(
            [f'{str(self.__class__)}(x={self.x},' , 
             f'y={self.y}', 
             f'secid={self.secid}', 
             f'date={self.date}', 
             f'feature={self.feature}', 
             f'weight_param={self.weight_param})'])

    def update_feature(self , use_feature = None):
        if use_feature is not None:
            assert all(np.isin(use_feature , self.feature)) , np.setdiff1d(use_feature , self.feature)
        self.use_feature = use_feature

    def SECID(self):
        return np.repeat(self.secid , len(self.date)).flatten()[self.finite.flatten()]

    def DATE(self):
        return np.tile(self.date , len(self.secid)).flatten()[self.finite.flatten()]

    def X(self): 
        if self.use_feature is None:
            return self.x.reshape(-1,self.x.shape[-1])[self.finite.flatten()]
        else:
            return self.X_feat(self.use_feature).reshape(-1,len(self.use_feature))[self.finite.flatten()]

    def Y(self): return self.y[self.finite]

    def W(self , weight_param : Optional[dict] = None):
        weight_param = self.weight_param if weight_param is None else weight_param
        if weight_param is None: weight_param = {}
        w = self.calculate_weight(self.y , **weight_param)
        return w[self.finite]
    
    def X_feat(self , feature): return self.x[...,match_values(feature , self.feature)]

    def reform_pred(self , pred):
        new_pred = self.y * 0
        new_pred[self.finite] = pred
        if self.input_type == pd.Series:
            new_pred = pd.DataFrame(new_pred , columns = self.date)
            new_pred[self.var_sec] = self.secid
            new_pred = new_pred.reset_index().melt(id_vars=self.var_sec,var_name=self.var_date)
            new_pred = new_pred.set_index([self.var_date,self.var_sec])['value'].loc[self.df_index]
        elif self.input_type == torch.Tensor:
            new_pred = torch.Tensor(new_pred)
        else:
            ...
        return pred

    @property
    def shape(self): return self.x.shape

    @property
    def nfeat(self): return len(self.feature) if self.use_feature is None else len(self.use_feature)
    
    @classmethod
    def calculate_weight(cls , y : np.ndarray, 
                         cs_type : Optional[Literal['top']] = 'top' ,
                         ts_type : Optional[Literal['lin' , 'exp']] = None ,
                         **kwargs):
        assert y.ndim == 2 or (y.ndim == 3 and y.shape[-1] == 1) , y.shape
        if y.ndim == 3: y = y[...,0]
        return cls.cs_weight(y , cs_type , **kwargs) * cls.ts_weight(y , ts_type , **kwargs)

    @classmethod
    def cs_weight(cls , y : np.ndarray , cs_type : Optional[Literal['top']] = 'top' , tau : Optional[float] = None , **kwargs):
        w = y * 0 + 1.
        if cs_type is None: return w
        if tau is None : tau = 0.75*np.log(0.5)/np.log(0.75)
        for j in range(w.shape[1]):
            if cs_type == 'top':
                v = y[:,j] * 1.
                v[~np.isnan(v)] = v[~np.isnan(v)].argsort()
                w[:,j] = np.exp((1 - v / np.nanmax(v))*np.log(0.5) / tau)
        return w
    
    @classmethod
    def ts_weight(cls , y : np.ndarray , ts_type : Optional[Literal['lin' , 'exp']] = None , rate : Optional[float] = None , **kwargs):
        w = y * 0 + 1.
        if ts_type is None: return w
        if rate is None : rate = 0.5
        if ts_type == 'lin':
            w *= np.linspace(rate,1,w.shape[1]).reshape(1,-1)
        elif ts_type == 'exp':
            w *= np.power(2 , -np.arange(w.shape[1])[::-1] / int(rate * w.shape[1])).reshape(1,-1)
        return w
