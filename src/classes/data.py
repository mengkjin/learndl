import numpy as np
import pandas as pd
import xarray as xr  
import torch

from copy import deepcopy
from dataclasses import dataclass
from torch import Tensor
from typing import Any , ClassVar , Literal , Optional

from ..func import match_values , index_union

@dataclass
class FailedData:
    type: str
    date: Optional[int] = None
    def add_attr(self , key , value): self.__dict__[key] = value

@dataclass(slots=True)
class NdData:
    values : np.ndarray | Tensor 
    index  : list[np.ndarray]
    def __post_init__(self):
        assert self.values.ndim == len(self.index) , (self.values.ndim , len(self.index))
        self.index = [(ii if isinstance(ii , np.ndarray) else np.array(ii)) for ii in self.index]

    def __repr__(self):
        return '\n'.join([str(self.__class__) , 
                          f'values shape : {self.shape}' , 
                          f'finite ratio : {self.finite_ratio():.4f}' , 
                          f'index : {str(self.index)}'])

    def __len__(self): return self.shape[0]

    @property
    def shape(self): return self.values.shape
    @property
    def ndim(self): return self.values.ndim

    def finite_ratio(self):
        if isinstance(self.values , np.ndarray):
            n_finite = np.isfinite(self.values).sum()
            n_total = self.values.size
        else:
            n_finite = self.values.isfinite().sum()
            n_total = self.values.numel()
        return n_finite / n_total

    @classmethod
    def from_xarray(cls , xarr : xr.Dataset):
        values = np.stack([arr.to_numpy() for arr in xarr.data_vars.values()] , -1)
        index = [arr.values for arr in xarr.indexes.values()] + [list(xarr.data_vars)]
        return cls(values , index)

    @classmethod
    def from_dataframe(cls , df : pd.DataFrame):
        index = [l.values for l in df.index.levels] + [df.columns.values] #type:ignore
        if len(df) != len(index[0]) * len(index[1]): 
            return cls.from_xarray(xr.Dataset.from_dataframe(df))
        else:
            values = df.values.reshape(len(index[0]) , len(index[1]) , -1)
            return cls(values , index)
        
@dataclass
class StockData4D:
    values  : Any = None 
    secid   : Any = None 
    date    : Any = None 
    feature : Any = None

    def __post_init__(self) -> None:
        if self.values is not None: 
            if isinstance(self.feature , str): 
                self.feature = np.array([self.feature])
            elif isinstance(self.feature , list):
                self.feature = np.array(self.feature)
            if isinstance(self.values , (int , float)):
                self.values = np.full((len(self.secid),len(self.date),1,len(self.feature)),self.values)
            if self.ndim == 3: self.values = self.values[:,:,None]
        self.asserted()

    def uninitiate(self):
        self.values  = None
        self.secid   = None
        self.date    = None
        self.feature = None

    def asserted(self):
        if self.shape:
            assert self.ndim == 4
            assert isinstance(self.values , (np.ndarray , Tensor))
            assert self.shape[0] == len(self.secid) 
            assert self.shape[1] == len(self.date)
            assert self.shape[3] == len(self.feature)
        return self
    
    def __repr__(self):
        if self.initiate:
            return '\n'.join(['initiated ' + str(self.__class__) , f'values shape {self.shape}'])
        else:
            return 'uninitiate ' + str(self.__class__) 
    @property
    def initiate(self): return self.values is not None
    @property
    def shape(self): return [] if self.values is None else self.values.shape 
    @property
    def dtype(self): return None if self.values is None else self.values.dtype
    @property
    def ndim(self): return None if self.values is None else self.values.ndim

    def update(self , **kwargs):
        [setattr(self,k,v) for k,v in kwargs.items() if k in ['values','secid','date','feature']]
        return self.asserted()
    
    def date_within(self , start : int , end : int , interval = 1) -> np.ndarray:
        return self.date[(self.date >= start) & (self.date <= end)][::interval]
    
    @classmethod
    def merge(cls , block_list):
        blocks = [blk for blk in block_list if isinstance(blk , cls) and blk.initiate]
        if len(blocks) == 0: return cls()
        elif len(blocks) == 1: return blocks[0]
            
        """
        secid  = index_union([blk.secid for blk in blocks])[0]
        date   = index_union([blk.date  for blk in blocks])[0]
        
        l1 = len(np.unique(np.concatenate([blk.feature for blk in blocks])))
        l2 = sum([len(blk.feature) for blk in blocks])
        distinct_feature = (l1 == l2)

        for blk in blocks: blk.align_secid_date(secid , date)
        values = [blk.values for blk in blocks]

        if distinct_feature:
            feature = np.concatenate([blk.feature for blk in blocks])
            newdata = np.concatenate([blk.values  for blk in blocks] , axis = -1)
        else:
            feature, p0f , p1f = index_union([blk.feature for blk in blocks])
            # print(feature, p0f , p1f)
            newdata = np.full((*blocks[0].shape[:-1],len(feature)) , np.nan , dtype = float)
            for i , data in enumerate(values): newdata[...,p0f[i]] = data[...,p1f[i]]

        return cls(newdata , secid , date , feature)
        
        """
        secid   , p0s , p1s = index_union([blk.secid   for blk in blocks])
        date    , p0d , p1d = index_union([blk.date    for blk in blocks])
        feature , p0f , p1f = index_union([blk.feature for blk in blocks])
        len_inday = blocks[0].shape[2]
        assert np.all([blk.shape[2] == len_inday for blk in blocks]) , 'blocks with different inday cannot be merged'
        p0i = p1i = np.arange(len_inday)

        for i , blk in enumerate(blocks): 
            if i == 0:
                new_blk = blocks[0].copy().align(secid , date , feature)
            else:
                new_blk.values[np.ix_(p0s[i],p0d[i],p0i,p0f[i])] = blk.values[np.ix_(p1s[i],p1d[i],p1i,p1f[i])]

        return new_blk

    def merge_others(self , others : list):
        return self.merge([self , *[others]])
    
    def as_tensor(self , asTensor = True):
        if asTensor and isinstance(self.values , np.ndarray): self.values = torch.tensor(self.values)
        return self
    
    def as_type(self , dtype = None):
        if dtype and isinstance(self.values , np.ndarray): self.values = self.values.astype(dtype)
        if dtype and isinstance(self.values , Tensor): self.values = self.values.to(dtype)
        return self
    
    def copy(self): return deepcopy(self)

    def align(self , secid = None , date = None , feature = None , inplace = True):
        obj = self if inplace else self.copy()
        obj = obj.align_secid_date(secid , date)
        obj = obj.align_feature(feature)
        return obj    

    def align_secid(self , secid , inplace = True):
        obj = self if inplace else self.copy()
        if secid is None or len(secid) == 0: return obj
        asTensor , dtype = isinstance(obj.values , Tensor) , obj.dtype
        values = np.full((len(secid) , *obj.shape[1:]) , np.nan)
        _ , p0s , p1s = np.intersect1d(secid , obj.secid , return_indices=True)
        values[p0s] = obj.values[p1s]
        obj.values = values
        obj.secid  = secid
        return obj.as_tensor(asTensor).as_type(dtype)
    
    def align_date(self , date , inplace = True):
        obj = self if inplace else self.copy()
        if date is None or len(date) == 0: return obj
        asTensor , dtype = isinstance(obj.values , Tensor) , obj.dtype
        values = np.full((obj.shape[0] , len(date) , *obj.shape[2:]) , np.nan)
        _ , p0d , p1d = np.intersect1d(date , obj.date , return_indices=True)
        values[:,p0d] = obj.values[:,p1d]
        obj.values  = values
        obj.date    = date
        return obj.as_tensor(asTensor).as_type(dtype)
    
    def align_secid_date(self , secid = None , date = None , inplace = True):
        obj = self if inplace else self.copy()
        if (secid is None or len(secid) == 0) and (date is None or len(date) == 0): 
            return obj
        elif secid is None or len(secid) == 0:
            return obj.align_date(date = date)
        elif date is None or len(date) == 0:
            return obj.align_secid(secid = secid)
        else:
            asTensor , dtype = isinstance(obj.values , Tensor) , obj.dtype
            values = np.full((len(secid),len(date),*obj.shape[2:]) , np.nan)
            _ , p0s , p1s = np.intersect1d(secid , obj.secid , return_indices=True)
            _ , p0d , p1d = np.intersect1d(date  , obj.date  , return_indices=True)
            values[np.ix_(p0s,p0d)] = obj.values[np.ix_(p1s,p1d)] 

            obj.values  = torch.tensor(values).to(obj.values) if isinstance(obj.values , Tensor) else values
            obj.secid   = secid
            obj.date    = date
            return obj.as_tensor(asTensor).as_type(dtype)
    
    def align_feature(self , feature , inplace = True):
        obj = self if inplace else self.copy()
        if feature is None or len(feature) == 0: return obj
        asTensor , dtype = isinstance(obj.values , Tensor) , obj.dtype
        values = np.full((*obj.shape[:-1],len(feature)) , np.nan)
        _ , p0f , p1f = np.intersect1d(feature , obj.feature , return_indices=True)
        values[...,p0f] = obj.values[...,p1f]
        obj.values  = torch.tensor(values).to(obj.values) if isinstance(obj.values , Tensor) else values
        obj.feature = feature
        return obj.as_tensor(asTensor).as_type(dtype)
    
    def add_feature(self , new_feature , new_value : np.ndarray | Tensor):
        assert new_value.shape == self.shape[:-1]
        new_value = new_value.reshape(*new_value.shape , 1)
        self.values  = np.concatenate([self.values,new_value],axis=-1)
        self.feature = np.concatenate([self.feature,[new_feature]],axis=0)
        return self
    
    def rename_feature(self , rename_dict : dict):
        if len(rename_dict) == 0: return self
        feature = self.feature.astype(object)
        for k,v in rename_dict.items(): feature[feature == k] = v
        self.feature = feature.astype(str)
        return self
    
    def loc(self , **kwargs) -> np.ndarray | Tensor:
        values : np.ndarray | Tensor = self.values
        for k,v in kwargs.items():  
            if isinstance(v , (str,int,float)): kwargs[k] = [v]
        if 'feature' in kwargs.keys(): 
            index  = match_values(kwargs['feature'] , self.feature)
            values = values[:,:,:,index]
        if 'inday'   in kwargs.keys(): 
            index  = match_values(kwargs['inday'] , range(values.shape[2]))
            values = values[:,:,index]
        if 'date'    in kwargs.keys(): 
            index  = match_values(kwargs['date'] , self.date)
            values = values[:,index]
        if 'secid'   in kwargs.keys(): 
            index  = match_values(kwargs['secid'] , self.secid)
            values = values[index]
        return values
    
    @classmethod
    def from_dataframe(cls , df : pd.DataFrame):
        xarr = NdData.from_xarray(xr.Dataset.from_dataframe(df))
        return cls(xarr.values , xarr.index[0] , xarr.index[1] , xarr.index[-1])
    
    def to_dataframe(self , drop_inday = True):
        df_dict = {}
        df_dict['secid'] = np.repeat(self.secid , self.shape[1] * self.shape[2])
        df_dict['date']  = np.repeat(np.tile(self.date , self.shape[0]) , self.shape[2])
        if not drop_inday or self.shape[2] > 1:
            df_dict['inday']  = np.tile(np.arange(self.shape[2]) , self.shape[0] * self.shape[1])
        
        df = pd.DataFrame({**df_dict , **{feat:self.loc(feature=feat).flatten() for feat in self.feature}}).set_index(['secid' , 'date'])
        return df

@dataclass
class BoosterData:
    raw_x   : Optional[pd.DataFrame | np.ndarray | torch.Tensor | NdData]
    raw_y   : Optional[pd.Series    | np.ndarray | torch.Tensor | NdData]
    secid   : Any = None
    date    : Any = None
    feature : Any = None
    weight_param : Optional[dict] = None

    VAR_SECID : ClassVar[list[str]] = ['SecID','instrument']
    VAR_DATE  : ClassVar[list[str]] = ['TradeDate','datetime']

    def __post_init__(self):
        x , y = self.raw_x , self.raw_y
        assert x is not None and y is not None
        assert len(x) == len(y) , f'x and y length must match'
        assert type(x) == type(y) , (f'x and y type must match')
        self.input_type = type(y)
        if isinstance(x , pd.DataFrame) and isinstance(y , pd.Series): 
            self.df_index = y.index
            self.var_sec  = [v for v in self.VAR_SECID  if v in x.index.names][0]
            self.var_date = [v for v in self.VAR_DATE if v in x.index.names][0]
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

        if x.ndim == 2: 
            x , y = x[:,None,:] , y[:,None]

        if self.secid is None:  self.secid = xindex[0] if xindex[0] is not None else np.arange(x.shape[0])
        if self.date  is None : self.date  = xindex[1] if xindex[1] is not None else np.arange(x.shape[1])
        if self.feature is None : self.feature = xindex[-1] if xindex[-1] is not None else np.array([f'F.{i}' for i in range(x.shape[-1])])

        assert x.shape == (len(self.secid) , len(self.date) , len(self.feature)) , (x.shape , (len(self.secid) , len(self.date) , len(self.feature)))
        assert y.shape == (len(self.secid) , len(self.date)) , (y.shape , (len(self.secid) , len(self.date)))

        self.x , self.y = x , y
        self.finite = np.isfinite(y)

        self.update_feature()
        if self.weight_param is None: 
            self.weight_param = {'tau':0.75*np.log(0.5)/np.log(0.75) , 'ts_type':'lin' , 'rate':0.5}  

        self.raw_x , self.raw_y = None , None

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
