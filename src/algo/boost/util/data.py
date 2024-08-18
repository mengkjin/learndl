import torch
import numpy as np
import pandas as pd
import xarray as xr

from dataclasses import dataclass , field
from typing import Any , Literal , Optional

from ....func import match_values

@dataclass
class BoosterOutput:
    pred    : np.ndarray | torch.Tensor
    secid   : np.ndarray
    date    : np.ndarray
    finite  : np.ndarray
    label   : np.ndarray | Any = None

    def to_2d(self):
        pred = self.label * 0
        pred[self.finite] = self.to_numpy()
        return pred

    def to_dataframe(self):
        df = pd.DataFrame(self.to_2d() , columns = self.date).assign(secid=self.secid).reset_index().\
            melt(id_vars='secid',var_name='date').set_index(['date','secid'])
        return df
    
    def to_numpy(self):
        return self.pred if isinstance(self.pred , np.ndarray) else self.pred.cpu().numpy()

@dataclass
class BoosterData:
    '''
    weight_param:
        ts_type : Literal['lin' , 'exp'] | None = None ,
        cs_type : Literal['top' , 'ones'] | None = None ,
        bm_type : Literal['in'] | None = None ,
        ts_lin_rate : float = 0.5 ,
        ts_half_life_rate : float = 0.5 ,
        cs_top_tau : float = 0.75*np.log(0.5)/np.log(0.75) ,
        cs_ones_rate : float = 2. ,
        bm_rate : float = 2. ,
        bm_secid : np.ndarray | list | None = None ,
    '''
    x : np.ndarray
    y : np.ndarray
    secid   : np.ndarray
    date    : np.ndarray
    feature : np.ndarray
    weight_param : dict[str,Any] = field(default_factory=dict)
    
    def __post_init__(self):
        assert self.x.shape == (len(self.secid) , len(self.date) , len(self.feature)) , \
            (self.x.shape , (len(self.secid) , len(self.date) , len(self.feature)))
        assert self.y.shape == (len(self.secid) , len(self.date)) , (self.y.shape , (len(self.secid) , len(self.date)))
        
        self.finite = np.isfinite(self.y)
        self.update_feature()
        self.weight_method = BoosterWeightMethod(**self.weight_param)

    def __repr__(self):
        return '\n'.join(
            [f'secid={self.secid}', 
             f'date={self.date}', 
             f'feature={self.feature}', 
             f'weight_method={self.weight_method})'])
    
    def set_weight_param(self , **weight_param):
        self.weight_method.reset(**weight_param)

    def update_feature(self , use_feature = None):
        if use_feature is None:
            self.use_feature = self.feature
        else:
            assert all(np.isin(use_feature , self.feature)) , np.setdiff1d(use_feature , self.feature)
            self.use_feature = use_feature

    def SECID(self): return np.repeat(self.secid , len(self.date))[self.finite.flatten()]

    def DATE(self): return np.tile(self.date , len(self.secid))[self.finite.flatten()]

    def X(self): return self.x[self.finite][...,self.feat_idx]

    def Y(self): return self.y[self.finite]

    def W(self): return self.weight_method.calculate_weight(self.y , self.secid)[self.finite]
    
    def XYW(self , as_tensor = False , device = None) -> tuple[Any,Any,Any]:
        x , y , w = self.X() , self.Y() , self.W()
        if as_tensor:
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            w = torch.tensor(w).to(device)
        return x, y, w
    
    def output(self , pred):
        return BoosterOutput(np.array(pred) , self.secid , self.date , self.finite , self.y)
    
    def output_reform(self , output : np.ndarray , reshape = True , as_dataframe = False):
        if reshape: 
            pred = self.y * 0
            pred[self.finite] = output
        else:
            pred = output
        
        if as_dataframe: 
            assert reshape , 'must reshape first'
            return pd.DataFrame(pred , columns = self.date).assign(secid=self.secid).reset_index().\
                melt(id_vars='secid',var_name='date').set_index(['date','secid'])
        else:
            return pred

    def reshape_pred(self , pred):
        new_pred = self.y * 0
        new_pred[self.finite] = pred
        return new_pred

    def pred_to_dataframe(self , pred):
        df = pd.DataFrame(pred , columns = self.date).assign(secid=self.secid).reset_index().\
            melt(id_vars='secid',var_name='date').set_index(['date','secid'])
        return df

    @property
    def feat_idx(self): return match_values(self.use_feature , self.feature)

    @property
    def shape(self): return self.x.shape

    @property
    def nfeat(self): return len(self.feature) if self.use_feature is None else len(self.use_feature)

    def to_dataframe(self):
        df = pd.DataFrame(self.X() , columns = self.feature)
        df['secid'] = self.SECID()
        df['date']  = self.DATE()
        df['label'] = self.Y()
        df = df.set_index(['secid' , 'date'])
        return df

    @classmethod
    def from_dataframe(cls , data : pd.DataFrame , weight_param : dict[str,Any] = {}):
        SECID_COLS = ['SecID','instrument','secid','StockID']
        DATE_COLS  = ['TradeDate','datetime','date']  
        if data.index.name or len(data.index.names) > 1: data = data.reset_index()

        var_sec  = np.intersect1d(SECID_COLS , data.columns.values)
        var_date = np.intersect1d(DATE_COLS  , data.columns.values)
        assert len(var_sec) == len(var_date) == 1, (var_sec , var_date , data.columns)
        data = data.set_index([var_sec[0] , var_date[0]])

        xarr = xr.Dataset.from_dataframe(data.iloc[:,:-1])
        xindex = [arr.values for arr in xarr.indexes.values()] + [list(xarr.data_vars)]
        x = np.stack([arr.to_numpy() for arr in xarr.data_vars.values()] , -1)

        yarr = xr.Dataset.from_dataframe(data.iloc[:,-1:])
        y = np.stack([arr.to_numpy() for arr in yarr.data_vars.values()] , -1)[...,0]
        
        secid , date , feature = xindex[0] , xindex[1] , xindex[-1]
        return cls(x , y , secid , date , feature , weight_param)

    @classmethod
    def from_numpy(cls , x : np.ndarray , y : np.ndarray , 
                   secid : Any = None , date : Any = None , feature : Any = None , 
                   weight_param : dict[str,Any] = {}):
        assert x.ndim in [2,3] , x.ndim
        assert y.ndim in [x.ndim - 1, x.ndim] , (y.ndim , x.ndim)
        if y.ndim == x.ndim:
            assert y.shape[-1] == 1 , f'Booster Data cannot deal with multilabels, but got {y.shape}'
            y = y[...,0]
        if x.ndim == 2: x , y = x[:,None,:] , y[:,None,:]

        if secid is None:  secid = np.arange(x.shape[0])
        if date  is None : date  = np.arange(x.shape[1])
        if feature is None : feature = np.array([f'F.{i}' for i in range(x.shape[-1])])

        return cls(x , y , secid , date , feature , weight_param)
    
    @classmethod
    def from_tensor(cls , x : torch.Tensor , y : torch.Tensor , 
                    secid : Any = None , date : Any = None , feature : Any = None , 
                    weight_param : dict[str,Any] = {}):
        return cls.from_numpy(x.detach().cpu().numpy() , y.detach().cpu().numpy() ,
                              secid , date , feature , weight_param)
    
    @classmethod
    def concat(cls , datas : list[Optional['BoosterData']]):
        real_datas = [data for data in datas if data is not None]
        if len(real_datas) <= 1: return real_datas[0]
        return cls.from_dataframe(pd.concat([data.to_dataframe() for data in real_datas]))
    
@dataclass(slots=True)
class BoosterWeightMethod:
    ts_type : Literal['lin' , 'exp'] | None = None
    cs_type : Literal['top' , 'positive' , 'ones'] | None = None
    bm_type : Literal['in'] | None = None
    ts_lin_rate : float = 0.5
    ts_half_life_rate : float = 0.5
    cs_top_tau : float = 0.75*np.log(0.5)/np.log(0.75)
    cs_ones_rate : float = 2.
    bm_rate : float = 2.
    bm_secid : np.ndarray | list | None = None

    def calculate_weight(self , y : np.ndarray, secid : Any):
        if y.ndim == 3 and y.shape[-1] == 1: y = y[...,0]
        assert y.ndim == 2 , y.shape
        return self.cs_weight(y) * self.ts_weight(y) * self.bm_weight(y , secid)

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
    
    def bm_weight(self , y : np.ndarray , secid : np.ndarray | list):
        w = y * 0 + 1.
        if self.bm_type is None: 
            return w
        elif self.bm_type == 'in': 
            if self.bm_secid is not None:
                w[np.isin(secid , self.bm_secid)] = 2 * w[np.isin(secid , self.bm_secid)]
        else:
            raise KeyError(self.bm_type)
        return w
    
    def reset(self , **kwargs):
        [setattr(self , k , v) for k,v in kwargs.items()]
