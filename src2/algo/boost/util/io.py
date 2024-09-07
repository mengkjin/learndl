
import torch
import numpy as np
import pandas as pd
import xarray as xr

from copy import deepcopy
from dataclasses import dataclass , field
from typing import Any , Literal , Optional

from ....func import match_values , index_union , rankic_2d , ic_2d

@dataclass
class BoosterOutput:
    pred    : torch.Tensor
    secid   : np.ndarray
    date    : np.ndarray
    finite  : torch.Tensor
    label   : torch.Tensor | Any = None

    def to_2d(self):
        pred = self.label.to(self.pred) * 0
        pred[self.finite] = self.pred
        return pred

    def to_dataframe(self):
        df = pd.DataFrame(self.to_2d().cpu().numpy() , columns = self.date).assign(secid=self.secid).reset_index().\
            melt(id_vars='secid',var_name='date').set_index(['date','secid'])
        return df
    
    def rankic(self):
        assert self.label is not None
        return rankic_2d(self.to_2d() , self.label , 0)
    
    def ic(self):
        assert self.label is not None
        return ic_2d(self.to_2d() , self.label , 0)

@dataclass
class BoosterInput:
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
    x : torch.Tensor
    y : torch.Tensor | Any = None
    w : torch.Tensor | Any = None
    secid   : np.ndarray | Any = None
    date    : np.ndarray | Any = None
    feature : np.ndarray | Any = None
    weight_param : dict[str,Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.update_feature()
        self.weight_method = BoosterWeightMethod(**self.weight_param)
        self.use_feature = self.feature

    def __repr__(self):
        return '\n'.join(
            [f'secid={self.secid}', 
             f'date={self.date}', 
             f'feature={self.feature}', 
             f'weight_method={self.weight_method})'])

    @property
    def complete(self):
        return self.y is not None and self.secid is not None and self.date is not None and self.feature is not None
    @property
    def finite(self):
        if self.y is not None:
            return self.y.isfinite()
        else:
            return torch.ones_like(self.x[:,:,0] , dtype=torch.bool)
        
    def choose_finite(self , obj : torch.Tensor | np.ndarray | None) -> Any:
        if obj is None: return obj
        if self.y is None: 
            if obj.ndim > 1: return obj.reshape(-1 , *obj.shape[2:])
            else: return obj
        else:
            finite = self.y.isfinite()
            if obj.ndim > 1: return obj[finite]
            else: return obj[finite.flatten()]
    
    def copy(self): return deepcopy(self)
    
    def set_weight_param(self , **weight_param):
        self.weight_method.reset(**weight_param)

    def update_feature(self , use_feature = None):
        if use_feature is None:
            self.use_feature = self.feature
        else:
            assert all(np.isin(use_feature , self.feature)) , np.setdiff1d(use_feature , self.feature)
            self.use_feature = use_feature

    def SECID(self , dropna = True): 
        secid = np.repeat(self.secid , len(self.date))
        return self.choose_finite(secid) if dropna else secid

    def DATE(self , dropna = True): 
        date = np.tile(self.date , len(self.secid))
        return self.choose_finite(date) if dropna else date

    def X(self): return self.choose_finite(self.x[...,self.feat_idx])
    def Y(self): return self.choose_finite(self.y)
    def W(self): 
        w = self.weight_method.calculate_weight(self.y , self.secid) if self.w is None and self.y is not None else self.w
        return self.choose_finite(w)
    def XYW(self , *args): return self.BoostXYW(self.X() , self.Y() , self.W())
    
    @dataclass
    class BoostXYW:
        x : Any
        y : Any
        w : Any
        
        def as_numpy(self):
            if isinstance(self.x , torch.Tensor): self.x = self.x.cpu().numpy()
            if isinstance(self.y , torch.Tensor): self.y = self.y.cpu().numpy()
            if isinstance(self.w , torch.Tensor): self.w = self.w.cpu().numpy()
            return self
    
    def output(self , pred : torch.Tensor | np.ndarray | Any):
        if isinstance(pred , torch.Tensor):
            new_pred = pred
        else:
            new_pred = torch.tensor(np.array(pred))
        finite = self.y.isfinite() if self.y is not None else torch.full_like(new_pred , fill_value=True)
        return BoosterOutput(new_pred , self.secid , self.date , finite , self.y)

    def pred_to_dataframe(self , pred : np.ndarray | torch.Tensor):
        new_pred = pred.numpy() if isinstance(pred , torch.Tensor) else pred 
        df = pd.DataFrame(new_pred , columns = self.date).assign(secid=self.secid).reset_index().\
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
        x = torch.tensor(np.stack([arr.to_numpy() for arr in xarr.data_vars.values()] , -1))

        yarr = xr.Dataset.from_dataframe(data.iloc[:,-1:])
        y = torch.tensor(np.stack([arr.to_numpy() for arr in yarr.data_vars.values()] , -1)[...,0])
        
        secid , date , feature = xindex[0] , xindex[1] , xindex[-1]
        return cls(x , y , None , secid , date , feature , weight_param)

    @classmethod
    def from_numpy(cls , x : np.ndarray , y : np.ndarray | Any = None,  w : np.ndarray | Any = None ,
                   secid : Any = None , date : Any = None , feature : Any = None , 
                   weight_param : dict[str,Any] = {}):
        return cls.from_tensor(torch.tensor(x) , None if y is None else torch.tensor(y) ,
                               None if w is None else torch.tensor(w) ,
                               secid , date , feature , weight_param)
    
    @classmethod
    def from_tensor(cls , x : torch.Tensor , y : torch.Tensor | Any = None , w : torch.Tensor | Any = None ,
                    secid : Any = None , date : Any = None , feature : Any = None , 
                    weight_param : dict[str,Any] = {}):
        assert x.ndim in [2,3] , x.ndim
        assert y is None or y.ndim in [x.ndim - 1, x.ndim] , (y.ndim , x.ndim)
        if y is not None and y.ndim == x.ndim:
            assert y.shape[-1] == 1 , f'Booster Data cannot deal with multilabels, but got {y.shape}'
            y = y[...,0]
        if y is not None and y.ndim == 1: y = y[:,None]
        if x.ndim == 2:  x = x[:,None,:]

        if secid is None:  secid = np.arange(x.shape[0])
        if date  is None : date  = np.arange(x.shape[1])
        if feature is None : feature = np.array([f'F.{i}' for i in range(x.shape[-1])])
        return cls(x , y , w , secid , date , feature , weight_param)
    
    @classmethod
    def concat(cls , datas : list[Optional['BoosterInput']]):
        blocks = [data for data in datas if data is not None and data.complete]
        
        secid   , p0s , p1s = index_union([blk.secid   for blk in blocks])
        date    , p0d , p1d = index_union([blk.date    for blk in blocks])
        feature , p0f , p1f = index_union([blk.feature for blk in blocks])
        
        x = np.full((len(secid) , len(date) , len(feature)) , fill_value=np.nan)
        y = np.full((len(secid) , len(date)) , fill_value=np.nan)

        for i , blk in enumerate(blocks): 
            x[np.ix_(p0s[i],p0d[i],p0f[i])] = blk.x[np.ix_(p1s[i],p1d[i],p1f[i])]
            y[np.ix_(p0s[i],p0d[i])]        = blk.y[np.ix_(p1s[i],p1d[i])]

        new_blk = cls(torch.tensor(x) , torch.tensor(y) , secid , date , feature)
        return new_blk
    
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

    def calculate_weight(self , y : np.ndarray | torch.Tensor , secid : Any):
        if y.ndim == 3 and y.shape[-1] == 1: y = y[...,0]
        assert y.ndim == 2 , y.shape
        return self.cs_weight(y) * self.ts_weight(y) * self.bm_weight(y , secid)

    def cs_weight(self , y : np.ndarray | torch.Tensor , **kwargs):
        w = y * 0 + 1.
        if self.cs_type is None: 
            return w
        elif self.cs_type == 'ones':
            w[y == 1.] = w[y == 1.] * 2
        elif self.cs_type == 'top':
            for j in range(w.shape[1]):
                v = y[:,j].argsort() + y[:,j] * 0
                w[:,j] = np.exp((1 - v / np.nanmax(v))*np.log(0.5) / self.cs_top_tau)
        else:
            raise KeyError(self.cs_type)
        return w
    
    def ts_weight(self , y : np.ndarray | torch.Tensor , **kwargs):
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
    
    def bm_weight(self , y : np.ndarray | torch.Tensor , secid : np.ndarray | list):
        w = y * 0 + 1.
        if self.bm_type is None: 
            return w
        elif self.bm_type == 'in': 
            if self.bm_secid is not None:
                w *= np.isin(secid , self.bm_secid) * 1 + 1
        else:
            raise KeyError(self.bm_type)
        return w
    
    def reset(self , **kwargs):
        [setattr(self , k , v) for k,v in kwargs.items()]
