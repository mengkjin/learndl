import numpy as np
import pandas as pd
import xarray as xr  
import torch

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
    index  : list | tuple
    def __post_init__(self):
        assert self.values.ndim == len(self.index) , (self.values.ndim , len(self.index))

    def __repr__(self):
        return '\n'.join([str(self.__class__) , f'values shape {self.shape}'])

    @property
    def shape(self): return self.values.shape
    @property
    def ndim(self): return self.values.ndim

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
            if self.ndim == 3: self.values = self.values[:,:,None]

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
            assert self.shape[4] == len(self.feature)
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
    
    @classmethod
    def merge(cls , block_list):
        blocks = [blk for blk in block_list if isinstance(blk , cls) and blk.initiate]
        if len(blocks) == 0: return cls()
        elif len(blocks) == 1: return blocks[0]
            
        values = [blk.values for blk in blocks]
        secid  = index_union([blk.secid for blk in blocks])[0]
        date   = index_union([blk.date  for blk in blocks])[0]
        l1 = len(np.unique(np.concatenate([blk.feature for blk in blocks])))
        l2 = sum([len(blk.feature) for blk in blocks])
        distinct_feature = (l1 == l2)

        for blk in blocks: blk.align_secid_date(secid , date)

        if distinct_feature:
            feature = np.concatenate([blk.feature for blk in blocks])
            newdata = np.concatenate([blk.values  for blk in blocks] , axis = -1)
        else:
            feature, p0f , p1f = index_union([blk.feature for blk in blocks])
            newdata = np.full((*blocks[0].shape[:-1],len(feature)) , np.nan , dtype = float)
            for i , data in enumerate(values): newdata[...,p0f[i]] = data[...,p1f[i]]

        return cls(newdata , secid , date , feature)

    def merge_others(self , others : list):
        return self.merge([self , *[others]])
    
    def as_tensor(self , asTensor = True):
        if asTensor and isinstance(self.values , np.ndarray): self.values = torch.tensor(self.values)
        return self
    
    def as_type(self , dtype = None):
        if dtype and isinstance(self.values , np.ndarray): self.values = self.values.astype(dtype)
        if dtype and isinstance(self.values , Tensor): self.values = self.values.to(dtype)
        return self
    
    def align(self , secid = None , date = None , feature = None):
        self = self.align_secid_date(secid , date)
        self = self.align_feature(feature)
        return self    

    def align_secid(self , secid):
        if secid is None or len(secid) == 0: return self
        asTensor , dtype = isinstance(self.values , Tensor) , self.dtype
        values = np.full((len(secid) , *self.shape[1:]) , np.nan)
        _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
        values[p0s] = self.values[p1s]
        self.values = values
        self.secid  = secid
        return self.as_tensor(asTensor).as_type(dtype)
    
    def align_date(self , date):
        if date is None or len(date) == 0: return self
        asTensor , dtype = isinstance(self.values , Tensor) , self.dtype
        values = np.full((self.shape[0] , len(date) , *self.shape[2:]) , np.nan)
        _ , p0d , p1d = np.intersect1d(date , self.date , return_indices=True)
        values[:,p0d] = self.values[:,p1d]
        self.values  = values
        self.date    = date
        return self.as_tensor(asTensor).as_type(dtype)
    
    def align_secid_date(self , secid = None , date = None):
        if (secid is None or len(secid) == 0) and (date is None or len(date) == 0): 
            return self
        elif secid is None or len(secid) == 0:
            return self.align_date(date = date)
        elif date is None or len(date) == 0:
            return self.align_secid(secid = secid)
        else:
            asTensor , dtype = isinstance(self.values , Tensor) , self.dtype
            values = np.full((len(secid),len(date),*self.shape[2:]) , np.nan)
            _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
            _ , p0d , p1d = np.intersect1d(date  , self.date  , return_indices=True)
            values[np.ix_(p0s,p0d)] = self.values[np.ix_(p1s,p1d)] 
            self.values  = torch.tensor(values).to(self.values) if isinstance(self.values , Tensor) else values
            self.secid   = secid
            self.date    = date
            return self.as_tensor(asTensor).as_type(dtype)
    
    def align_feature(self , feature):
        if feature is None or len(feature) == 0: return self
        asTensor , dtype = isinstance(self.values , Tensor) , self.dtype
        values = np.full((*self.shape[:-1],len(feature)) , np.nan)
        _ , p0f , p1f = np.intersect1d(feature , self.feature , return_indices=True)
        values[...,p0f] = self.values[...,p1f]
        self.values  = torch.tensor(values).to(self.values) if isinstance(self.values , Tensor) else values
        self.feature = feature
        return self.as_tensor(asTensor).as_type(dtype)
    
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

@dataclass
class BoosterData:
    raw_x   : pd.DataFrame | np.ndarray | torch.Tensor
    raw_y   : pd.Series    | np.ndarray | torch.Tensor
    secid   : Any = None
    date    : Any = None
    feature : Any = None
    weight_param : Optional[dict] = None

    df_var_sec  : ClassVar[list[str]] = ['SecID','instrument']
    df_var_date : ClassVar[list[str]] = ['TradeDate','datetime']

    def __post_init__(self):
        assert len(self.raw_x) == len(self.raw_y) , f'x and y length must match'
        if isinstance(self.raw_x , torch.Tensor) and isinstance(self.raw_y , torch.Tensor): 
            self.x = self.raw_x.detach().cpu().numpy()
            self.y = self.raw_y.detach().cpu().numpy()
        elif isinstance(self.raw_x , np.ndarray) and isinstance(self.raw_y , np.ndarray): 
            self.x = self.raw_x
            self.y = self.raw_y
        elif isinstance(self.raw_x , pd.DataFrame) and isinstance(self.raw_y , pd.Series): 
            self.var_sec  = [v for v in self.df_var_sec  if v in self.raw_x.index.names][0]
            self.var_date = [v for v in self.df_var_date if v in self.raw_x.index.names][0]
            x = self.raw_x.reset_index().set_index([self.var_sec,self.var_date])
            xarr = xr.Dataset.from_dataframe(x)
 
            xindex = [arr.values for arr in xarr.indexes.values()] + [list(xarr.data_vars)]
            self.x = np.stack([arr.to_numpy() for arr in xarr.data_vars.values()] , -1)
            if self.secid is None : self.secid = xindex[0]
            if self.date  is None : self.date  = xindex[1]
            if self.feature is None : self.feature = xindex[-1]

            yarr = xr.Dataset.from_dataframe(pd.DataFrame(self.raw_y.reset_index().set_index([self.var_sec,self.var_date])))
            self.y = np.stack([arr.to_numpy() for arr in yarr.data_vars.values()] , -1)
        else:
            raise TypeError(f'x and y type must match')
        if self.y.ndim == 3:
            assert self.y.shape[-1] == 1
            self.y = self.y[...,0]
        self.finite = np.isfinite(self.y)
        if self.secid is None : self.secid = np.arange(self.x.shape[0])
        if self.date  is None : self.date  = np.arange(self.x.shape[1])
        if self.feature is None : self.feature = np.array([f'feature.{i}' for i in range(self.x.shape[-1])])
        assert self.x.shape == (len(self.secid) , len(self.date) , len(self.feature))
        assert self.y.shape == (len(self.secid) , len(self.date))
        self.update_feature()
        if self.weight_param is None: self.weight_param = {'tau':0.75*np.log(0.5)/np.log(0.75) , 'ts_type':'lin' , 'rate':0.5}  

    def update_feature(self , use_feature = None):
        if use_feature is not None:
            assert all(np.isin(use_feature , self.feature)) , np.setdiff1d(use_feature , self.feature)
        self.use_feature = use_feature

    def X(self): 
        if self.use_feature is None:
            return self.x.reshape(-1,self.x.shape[-1])[self.finite.flatten()]
        else:
            return self.X_feat(self.use_feature).reshape(-1,len(self.use_feature))[self.finite.flatten()]

    def Y(self): return self.y.flatten()[self.finite.flatten()]

    def W(self , weight_param : Optional[dict] = None):
        weight_param = self.weight_param if weight_param is None else weight_param
        if weight_param is None: weight_param = {}
        w = self.calculate_weight(self.y , **weight_param)
        return w.flatten()[self.finite.flatten()]
    
    def X_feat(self , feature): return self.x[...,match_values(feature , self.feature)]

    def reform_pred(self , pred):
        new_pred = self.y.flatten()
        new_pred[self.finite.flatten()] = pred
        pred = new_pred.reshape(*self.y.shape)
        pred = np.array(pred)
        if isinstance(self.raw_y , pd.Series):
            pred = pd.DataFrame(pred , columns = self.date)
            pred[self.var_sec] = self.secid
            pred = pred.reset_index().melt(id_vars=self.var_sec,var_name=self.var_date)
            pred = pred.set_index([self.var_date,self.var_sec])['value'].loc[self.raw_y.index]
        elif isinstance(self.raw_y , torch.Tensor):
            pred = torch.Tensor(pred)
            ...
        else:
            if not isinstance(pred , np.ndarray): pred = np.array(pred)
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
