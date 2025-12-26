import torch
import numpy as np
import pandas as pd
import xarray as xr  

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.proj import Logger
from src.math import match_values , index_union , index_stack , to_numpy

from .nd import NdData


@dataclass
class Stock4DData:
    values  : Any = None 
    secid   : Any = None 
    date    : Any = None 
    feature : Any = None

    def __post_init__(self) -> None:
        if isinstance(self.secid , (int , float , str)):
            self.secid = np.array([self.secid])
        if isinstance(self.date , (int , float , str)):
            self.date = np.array([self.date])
        if isinstance(self.feature , (int , float , str)):
            self.feature = np.array([self.feature])
        if self.values is not None: 
            if isinstance(self.feature , str): 
                self.feature = np.array([self.feature])
            elif isinstance(self.feature , list):
                self.feature = np.array(self.feature)
            if isinstance(self.values , (int , float)):
                self.values = np.full((len(self.secid),len(self.date),1,len(self.feature)),self.values)
            if self.ndim == 3: 
                self.values = self.values[:,:,None]
        self.asserted()

    def uninitiate(self):
        self.values  = None
        self.secid   = None
        self.date    = None
        self.feature = None

    def asserted(self):
        if self.initiate:
            assert isinstance(self.values , (np.ndarray , torch.Tensor)) , self.values
            assert self.ndim == 4 , self.shape
            assert self.shape[0] == len(self.secid) , (self.shape[0] , len(self.secid))
            assert self.shape[1] == len(self.date) , (self.shape[1] , len(self.date))
            assert self.shape[3] == len(self.feature) , (self.shape[3] , len(self.feature))
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
    @property
    def empty(self): return self.values is None or self.values.size == 0
    @property
    def dates(self): return self.date if not self.empty else np.array([])
    @property
    def max_date(self): return -1 if self.empty else self.date.max()
    @property
    def min_date(self): return 99991231 if self.empty else self.date.min()

    def update(self , **kwargs):
        if 'values' in kwargs:
            self.values = kwargs['values']
        if 'secid' in kwargs:
            self.secid = to_numpy(kwargs['secid'])
        if 'date' in kwargs:
            self.date = to_numpy(kwargs['date'])
        if 'feature' in kwargs:
            self.feature = to_numpy(kwargs['feature'])
        return self.asserted()
    
    def date_within(self , start : int , end : int , interval = 1) -> np.ndarray:
        return self.date[(self.date >= start) & (self.date <= end)][::interval]
    
    @classmethod
    def merge(cls , block_list , inplace = False):
        """merge multiple blocks into one block , if inplace is True, merge into the first block"""
        blocks = [blk for blk in block_list if isinstance(blk , cls) and not blk.empty]
        if len(blocks) == 0: 
            return cls()
        elif len(blocks) == 1: 
            return blocks[0] if inplace else blocks[0].copy()
            
        secid   , p0s , p1s = index_union([blk.secid   for blk in blocks])
        date    , p0d , p1d = index_union([blk.date    for blk in blocks])
        feature , p0f , p1f = index_stack([blk.feature for blk in blocks])
        len_inday = blocks[0].shape[2]
        assert np.all([blk.shape[2] == len_inday for blk in blocks]) , 'blocks with different inday cannot be merged'
        p0i = p1i = np.arange(len_inday)

        new_blk = blocks[0].align(secid , date , feature , inplace = inplace)
        for i , blk in enumerate(blocks[1:] , start = 1): 
            new_blk.values[np.ix_(p0s[i],p0d[i],p0i,p0f[i])] = blk.values[np.ix_(p1s[i],p1d[i],p1i,p1f[i])]

        return new_blk

    def merge_others(self , others : list | Any , inplace = False):
        if not isinstance(others , list): 
            others = [others]
        return self.merge([self , *others] , inplace = inplace).align_feature(self.feature , inplace = True)
    
    def as_tensor(self , asTensor = True):
        if asTensor and isinstance(self.values , np.ndarray): 
            self.values = torch.from_numpy(self.values)
        return self
    
    def as_type(self , dtype = None):
        if dtype and isinstance(self.values , np.ndarray): 
            self.values = self.values.astype(dtype)
        if dtype and isinstance(self.values , torch.Tensor): 
            self.values = self.values.to(dtype)
        return self
    
    def copy(self): return deepcopy(self)

    def align(self , secid = None , date = None , feature = None , inplace = False):
        blk = self.align_secid_date(secid , date , inplace = inplace)
        blk = blk.align_feature(feature , inplace = True)
        return blk

    def align_secid(self , secid , inplace = False):
        secid = None if secid is None else to_numpy(secid)
        if secid is None or len(secid) == 0: 
            return self if inplace else self.copy()
        asTensor , dtype = isinstance(self.values , torch.Tensor) , self.dtype
        values = np.full((len(secid) , *self.shape[1:]) , np.nan)
        _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
        values[p0s] = self.values[p1s]
        values = torch.tensor(values).to(self.values) if asTensor else values
        if inplace:
            return self.update(values = values , secid = secid).as_type(dtype)
        else:
            return self.__class__(values = values , secid = secid , date = self.date , feature = self.feature).as_type(dtype)

    def align_date(self , date , inplace = False):
        date = None if date is None else to_numpy(date)
        if date is None or len(date) == 0: 
            return self if inplace else self.copy()
        asTensor , dtype = isinstance(self.values , torch.Tensor) , self.dtype
        values = np.full((self.shape[0] , len(date) , *self.shape[2:]) , np.nan)
        _ , p0d , p1d = np.intersect1d(date , self.date , return_indices=True)
        values[:,p0d] = self.values[:,p1d]
        values  = torch.tensor(values).to(self.values) if asTensor else values
        if inplace:
            return self.update(values = values , date = date).as_type(dtype)
        else:
            return self.__class__(values = values , secid = self.secid , date = date , feature = self.feature).as_type(dtype)
    
    def align_secid_date(self , secid = None , date = None , inplace = False):
        secid = None if secid is None else to_numpy(secid)
        date = None if date is None else to_numpy(date)
        if (secid is None or len(secid) == 0) and (date is None or len(date) == 0): 
            return self if inplace else self.copy()
        elif secid is None or len(secid) == 0:
            return self.align_date(date = date, inplace = inplace)
        elif date is None or len(date) == 0:
            return self.align_secid(secid = secid, inplace = inplace)
        else:
            asTensor , dtype = isinstance(self.values , torch.Tensor) , self.dtype
            values = np.full((len(secid),len(date),*self.shape[2:]) , np.nan)
            _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
            _ , p0d , p1d = np.intersect1d(date  , self.date  , return_indices=True)
            values[np.ix_(p0s,p0d)] = self.values[np.ix_(p1s,p1d)] 

            values = torch.tensor(values).to(self.values) if asTensor else values

            if inplace:
                return self.update(values = values , secid = secid , date = date).as_type(dtype)
            else:
                return self.__class__(values = values , secid = secid , date = date , feature = self.feature).as_type(dtype)
    
    def align_feature(self , feature , inplace = False):
        feature = None if feature is None else to_numpy(feature) 
        if feature is None or len(feature) == 0: 
            return self if inplace else self.copy()
        asTensor , dtype = isinstance(self.values , torch.Tensor) , self.dtype
        values = np.full((*self.shape[:-1],len(feature)) , np.nan)
        _ , p0f , p1f = np.intersect1d(feature , self.feature , return_indices=True)
        values[...,p0f] = self.values[...,p1f]
        values  = torch.tensor(values).to(self.values) if asTensor else values
        if inplace:
            return self.update(values = values , feature = feature).as_type(dtype)
        else:
            return self.__class__(values = values , secid = self.secid , date = self.date , feature = feature).as_type(dtype)
    
    def add_feature(self , new_feature , new_value : np.ndarray | torch.Tensor):
        assert new_value.shape == self.shape[:-1] , (new_value.shape , self.shape[:-1])
        new_value = new_value.reshape(*new_value.shape , 1)
        self.values  = np.concatenate([self.values,new_value],axis=-1)
        self.feature = np.concatenate([self.feature,[new_feature]],axis=0)
        return self
    
    def rename_feature(self , rename_dict : dict):
        if len(rename_dict) == 0: 
            return self
        feature = self.feature.astype(object)
        for k,v in rename_dict.items(): 
            feature[feature == k] = v
        self.feature = feature.astype(str)
        return self
    
    def loc(self , secid : Any | None = None , date : Any | None = None , feature : Any | None = None , inday : Any | None = None , fillna : Any = None):
        values : np.ndarray | torch.Tensor | Any = self.values

        if feature is not None: 
            index  = match_values(feature , self.feature)
            values = values[:,:,:,index]
        if inday is not None: 
            index  = match_values(inday , range(values.shape[2]))
            values = values[:,:,index]
        if date is not None: 
            index  = match_values(date , self.date)
            values = values[:,index]
        if secid is not None: 
            index  = match_values(secid , self.secid)
            values = values[index,:]
        if fillna is not None: 
            if isinstance(values , torch.Tensor): 
                values = values.nan_to_num(fillna)
            elif isinstance(values , np.ndarray): 
                values = np.nan_to_num(values,fillna)
            else:
                raise TypeError(f'Unsupported type: {type(values)} for {self.__class__.__name__} values')
        return values


    @classmethod
    def concat_feature(cls , block_list):
        blocks = [blk for blk in block_list if isinstance(blk , cls) and blk.initiate] 
        for i , blk in enumerate(blocks): 
            if i == 0:
                new_blk = blk.copy()
            else:
                assert np.array_equal(new_blk.secid , blk.secid) , (new_blk.secid , blk.secid)
                assert np.array_equal(new_blk.date , blk.date) , (new_blk.date , blk.date)
                new_blk.feature = np.concatenate([new_blk.feature , blk.feature])
                new_blk.values  = np.concatenate([new_blk.values  , blk.values ] , axis=-1)
        return new_blk
    
    @classmethod
    def from_dataframe(cls , df : pd.DataFrame | None):
        if df is None or df.empty: 
            return cls()
        try:
            xarr = NdData.from_xarray(xr.Dataset.from_dataframe(df))
        except Exception as e:
            Logger.error(e)
            Logger.stdout(df[df.index.duplicated()])
            raise e
        try:
            value = cls(xarr.values , xarr.index[0] , xarr.index[1] , xarr.index[-1])
        except:
            import src
            setattr(src , 'xarr' , xarr)
            Logger.stdout(xarr)
            raise
        return value

        # return cls(xarr.values , xarr.index[0] , xarr.index[1] , xarr.index[-1])
    
    def to_dataframe(self , drop_inday = True):
        df_dict = {}
        df_dict['secid'] = np.repeat(self.secid , self.shape[1] * self.shape[2])
        df_dict['date']  = np.repeat(np.tile(self.date , self.shape[0]) , self.shape[2])
        if not drop_inday or self.shape[2] > 1:
            df_dict['inday']  = np.tile(np.arange(self.shape[2]) , self.shape[0] * self.shape[1])
        
        df = pd.DataFrame(df_dict  | {feat:self.loc(feature=feat).flatten() for feat in self.feature}).set_index(['secid' , 'date'])
        return df