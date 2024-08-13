import torch
import numpy as np
import pandas as pd
import xarray as xr

from dataclasses import dataclass
from typing import Any , ClassVar , Literal , Optional

from .nd import NdData
from ...func import match_values

@dataclass(slots=True)
class WeightMethod:
    ts_type : Literal['lin' , 'exp'] | None = None
    cs_type : Literal['top' , 'positive' , 'ones'] | None = None
    bm_type : Literal['in'] | None = None
    ts_lin_rate : float = 0.5
    ts_half_life_rate : float = 0.5
    cs_top_tau : float = 0.75*np.log(0.5)/np.log(0.75)
    cs_ones_rate : float = 2.
    bm_rate : float = 2.

    def calculate_weight(self , y : np.ndarray, secid : Any , bm_secid : Any):
        if y.ndim == 3 and y.shape[-1] == 1: y = y[...,0]
        assert y.ndim == 2 , y.shape
        return self.cs_weight(y) * self.ts_weight(y) * self.bm_weight(y , secid , bm_secid)

    def cs_weight(self , y : np.ndarray , **kwargs):
        w = y * 0 + 1.
        if self.cs_type is None: 
            return w
        elif self.cs_type == 'ones':
            w[y == 1.] = w[y == 1.] * 2
        elif self.cs_type == 'top':
            for j in range(w.shape[1]):
                v = y[:,j] * 1.
                v[~np.isnan(v)] = v[~np.isnan(v)].argsort()
                w[:,j] = np.exp((1 - v / np.nanmax(v))*np.log(0.5) / self.cs_top_tau)
        else:
            raise KeyError(self.cs_type)
        return w
    
    def ts_weight(self , y : np.ndarray , **kwargs):
        w = y * 0 + 1.
        if self.ts_type is None: 
            return w
        elif self.ts_type == 'lin':
            w *= np.linspace(self.ts_lin_rate,1,w.shape[1]).reshape(1,-1)
        elif self.ts_type == 'exp':
            w *= np.power(2 , -np.arange(w.shape[1])[::-1] / int(self.ts_half_life_rate * w.shape[1])).reshape(1,-1)
        else:
            raise KeyError(self.ts_type)
        return w
    
    def bm_weight(self , y : np.ndarray , secid : np.ndarray | list , bm_secid : np.ndarray | list):
        w = y * 0 + 1.
        if self.bm_type is None: 
            return w
        elif self.bm_type == 'in': 
            w[np.isin(secid , bm_secid)] = 2 * w[np.isin(secid , bm_secid)]
        else:
            raise KeyError(self.bm_type)
        return w
    
    def reset(self , **kwargs):
        [setattr(self , k , v) for k,v in kwargs.items()]

class BoosterData:
    SECID_COLS : ClassVar[list[str]] = ['SecID','instrument','secid','StockID']
    DATE_COLS  : ClassVar[list[str]] = ['TradeDate','datetime','date']

    def __init__(
        self , 
        x : pd.DataFrame | np.ndarray | torch.Tensor | NdData ,
        y : pd.Series    | np.ndarray | torch.Tensor | NdData ,
        secid   : Any = None ,
        date    : Any = None ,
        feature : Any = None ,
        ts_type : Literal['lin' , 'exp'] | None = None ,
        cs_type : Literal['top' , 'ones'] | None = None ,
        bm_type : Literal['in'] | None = None ,
        ts_lin_rate : float = 0.5 ,
        ts_half_life_rate : float = 0.5 ,
        cs_top_tau : float = 0.75*np.log(0.5)/np.log(0.75) ,
        cs_ones_rate : float = 2. ,
        bm_rate : float = 2. ,
        bm_secid : np.ndarray | list | None = None ,
    ):
        assert len(x) == len(y) , f'x and y length must match'
        if isinstance(x , pd.DataFrame) and isinstance(y , (pd.DataFrame , pd.Series)): 
            self.input_type = 'DataFrame'
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
            self.input_type = 'NdData'
            xindex = x.index
            x = x.values
            y = y.values
        elif isinstance(x , torch.Tensor) and isinstance(y , torch.Tensor):
            self.input_type = 'Tensor'
            xindex = [None , None , None]
        elif isinstance(x , np.ndarray) and isinstance(y , np.ndarray):
            self.input_type = 'array'
            xindex = [None , None , None]
        else:
            raise TypeError(type(x) , type(y))

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
        self.weight_method = WeightMethod(
            ts_type , cs_type , bm_type , ts_lin_rate , ts_half_life_rate , cs_top_tau ,
            cs_ones_rate , bm_rate)

        self.secid , self.date , self.feature = secid , date , feature
        self.bm_secid = bm_secid

    def __repr__(self):
        return '\n'.join(
            [f'{str(self.__class__)}(x={self.x},' , 
             f'y={self.y}', 
             f'secid={self.secid}', 
             f'date={self.date}', 
             f'feature={self.feature}', 
             f'weight_method={self.weight_method})'])
    
    def set_weight_param(self , **kwargs):
        self.weight_method.reset(**kwargs)

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

    def Y(self): 
        return self.y[self.finite]

    def W(self):
        w = self.weight_method.calculate_weight(self.y , self.secid , self.bm_secid)
        return w[self.finite]
    
    def XYW(self , as_tensor = False , device = None):
        x , y , w = self.X() , self.Y() , self.W()
        if as_tensor:
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            w = torch.tensor(w).to(device)
        return x, y, w
    
    def X_feat(self , feature): return self.x[...,match_values(feature , self.feature)]

    def reshape_pred(self , pred):
        new_pred = self.y * 0
        new_pred[self.finite] = pred
        return new_pred

    def reform_pred(self , pred):
        if self.input_type == 'DataFrame':
            pred = pd.DataFrame(pred , columns = self.date)
            pred[self.var_sec] = self.secid
            pred = pred.reset_index().melt(id_vars=self.var_sec,var_name=self.var_date)
            pred = pred.set_index([self.var_date,self.var_sec])['value'].loc[self.df_index]
        elif self.input_type == 'Tensor':
            pred = torch.tensor(pred)
        return pred

    @property
    def shape(self): return self.x.shape

    @property
    def nfeat(self): return len(self.feature) if self.use_feature is None else len(self.use_feature)