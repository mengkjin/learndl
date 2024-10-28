
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
    x: 3d tensor, (n_sample, n_date, n_feature)
    y: 2d tensor, (n_sample, n_date)    
    w: 2d tensor, (n_sample, n_date)
    secid: 1d array, (n_sample)
    date: 1d array, (n_date)
    feature: 1d array, (n_feature)
    weight_param:
        ts_type : Literal['lin' , 'exp'] | None = None ,
        cs_type : Literal['top' , 'ones'] | None = None ,
        bm_type : Literal['in'] | None = None ,
        ts_lin_rate : float = 0.5 ,
        ts_half_life_rate : float = 0.5 ,
        cs_top_tau : float = 0.75*np.log(0.5)/np.log(0.75) ,
        cs_ones_rate : float = 2. ,
        bm_rate : float = 2. ,
        bm_secid : np.ndarray | list | None = None
    as_categorical: whether to convert y to categorical
    '''
    x : torch.Tensor
    y : torch.Tensor | Any = None
    w : torch.Tensor | Any = None
    secid   : np.ndarray | Any = None
    date    : np.ndarray | Any = None
    feature : np.ndarray | Any = None
    weight_param : dict[str,Any] = field(default_factory=dict)
    n_bins : int | None = None
    
    def __post_init__(self):
        self.update_feature()
        self.weight_method = BoosterWeightMethod(**self.weight_param)
        self.use_feature = self.feature
        self._raw_y = self.y
        self.to_categorical(n_bins=self.n_bins)

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
            return self.y >= 0 if self.is_categorical else self.y.isfinite()
        else:
            return torch.ones_like(self.x[:,:,0] , dtype=torch.bool)
    @property
    def is_categorical(self): return self.n_bins is not None

    def copy(self): return deepcopy(self)

    def to_categorical(self , n_bins : int | None = 100):
        if n_bins is None:
            self.n_bins = None
            self.y = self._raw_y
        elif n_bins is not None and self.n_bins != n_bins:
            rank_pct = self._raw_y.argsort(dim = 0).argsort(dim = 0).where(~self._raw_y.isnan() , torch.nan)
            rank_pct /= rank_pct.nan_to_num().max(dim = 0 , keepdim = True)[0] + 1e-6
            self.y = (rank_pct * n_bins).int().clip(-1 , n_bins-1)
            self.n_bins = n_bins

        return self

    def set_weight_param(self , **weight_param):
        self.weight_method.reset(**weight_param)

    def update_feature(self , use_feature = None):
        if use_feature is None:
            self.use_feature = self.feature
        else:
            assert all(np.isin(use_feature , self.feature)) , np.setdiff1d(use_feature , self.feature)
            self.use_feature = use_feature

    def obj_flatten(self , obj : torch.Tensor | np.ndarray | None , dropna = True , date_first = True) -> Any:
        if obj is None: return obj

        if dropna and self.y is not None:
            finite = self.finite
            if date_first: finite = finite.transpose(1,0)
            if obj.ndim == 1: finite = finite.flatten()
        else:
            finite = None

        if date_first:
            obj = obj.transpose(1,0) if isinstance(obj , torch.Tensor) else obj.swapaxes(1,0)

        return obj[finite] if finite is not None else obj

    def SECID(self , dropna = True):
        return self.obj_flatten(self.secid[:,None].repeat(len(self.date),axis=1) , dropna=dropna)

    def DATE(self , dropna = True): 
        return self.obj_flatten(self.date[None,:].repeat(len(self.secid),axis=0) , dropna=dropna)

    def X(self) -> torch.Tensor: 
        return self.obj_flatten(self.x[...,self.feat_idx] , dropna=True)
    
    def Y(self): 
        return self.obj_flatten(self.y , dropna=True)
    
    def W(self): 
        w = self.weight_method.calculate_weight(self.y , self.secid) if self.w is None and self.y is not None else self.w
        return self.obj_flatten(w , dropna=True)
    
    def Dataset(self , *args): 
        return self.BoostDataset(self.X() , self.Y() , self.W() , self.DATE())
    @dataclass(slots=True)
    class BoostDataset:
        x : Any
        y : Any
        w : Any
        date : Any = None
        
        def as_numpy(self):
            [setattr(self , attr , getattr(self , attr).cpu().numpy()) for attr in self.__slots__ 
             if isinstance(getattr(self , attr) , torch.Tensor)]
            return self

        def to(self , device : torch.device | None = None):
            [setattr(self , attr , getattr(self , attr).to(device)) for attr in self.__slots__ 
             if isinstance(getattr(self , attr) , torch.Tensor)]
            return self
        
        def lgbm_inputs(self):
            self.as_numpy()
            return {'data' : self.x , 'label' : self.y , 'weight' : self.w , 'group' : self.group_arr()}
        def catboost_inputs(self): 
            self.as_numpy()
            return {'data' : self.x , 'label' : self.y , 'weight' : self.w , 'group_id' : self.group_id()}
        def xgboost_inputs(self): 
            self.as_numpy()
            return {'data' : self.x , 'label' : self.y , 'weight' : self.w , 'group' : self.group_arr()}
        
        def group_arr(self):
            assert (np.diff(self.date) >= 0).all() , 'date must be sorted'
            return np.unique(self.date , return_counts=True)[1]
        
        def group_id(self): return self.date

        @property
        def nfeat(self): return self.x.shape[-1]
        
    def output(self , pred : torch.Tensor | np.ndarray | Any):
        if isinstance(pred , torch.Tensor):
            new_pred = pred
        else:
            new_pred = torch.tensor(np.array(pred))

        if new_pred.ndim == 2: 
            weight = torch.arange(new_pred.shape[1]).to(new_pred) - (new_pred.shape[1] - 1) / 2
            new_pred = new_pred @ weight
        elif new_pred.ndim > 2:
            raise ValueError(f'BoosterOutput cannot deal with pred with ndim {new_pred.ndim}')

        finite = self.finite if self.y is not None else torch.full_like(new_pred , fill_value=True)
        return BoosterOutput(new_pred , self.secid , self.date , finite , self._raw_y)

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
        df = pd.DataFrame(self.X().cpu().numpy() , columns = self.feature)
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
        w = None if all(blk.w is None for blk in blocks) else np.ones_like(y)
        for i , blk in enumerate(blocks): 
            x[np.ix_(p0s[i],p0d[i],p0f[i])] = blk.x[np.ix_(p1s[i],p1d[i],p1f[i])]
            y[np.ix_(p0s[i],p0d[i])]        = blk.y[np.ix_(p1s[i],p1d[i])]
            if blk.w is not None and w is not None:
                w[np.ix_(p0s[i],p0d[i])]    = blk.w[np.ix_(p1s[i],p1d[i])]

        new_blk = cls(torch.tensor(x) , torch.tensor(y) , None if w is None else torch.tensor(w) , secid , date , feature)
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
