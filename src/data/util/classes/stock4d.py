from __future__ import annotations

import torch
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr  

from copy import deepcopy
from dataclasses import dataclass
from typing import Any , Literal

from src.proj import Logger
from src.proj.util import properties
from src.func import match_slice , index_merge , intersect_meshgrid , intersect_pos_slice

from .nd import NdData

INDAY_MARK_COLUMNS = ('inday' , 'minute')

@dataclass
class Stock4D:
    """four-dimensional data, (secid, date, inday, feature)"""
    values  : torch.Tensor | Any = None 
    secid   : np.ndarray | Any = None 
    date    : np.ndarray | Any = None 
    feature : np.ndarray | Any = None

    def __post_init__(self) -> None:
        self.asserted()

    def uninitiate(self):
        self.values  = None
        self.secid   = None
        self.date    = None
        self.feature = None

    @classmethod
    def as_array(cls , values : np.ndarray | torch.Tensor | list | tuple | str | int | float | Any) -> np.ndarray:
        if isinstance(values , np.ndarray):
            return values
        elif isinstance(values , torch.Tensor):
            return values.cpu().numpy()
        elif isinstance(values , (list , tuple)):
            return np.asarray(values)
        elif isinstance(values , (int , float , str)):
            return np.asarray([values])
        else:
            raise ValueError(f'Unsupported type: {type(values)} for {cls.__name__} values')

    def update(self , **kwargs):
        if 'values' in kwargs:
            self.values = kwargs['values']
        if 'secid' in kwargs:
            self.secid = kwargs['secid']
        if 'date' in kwargs:
            self.date = kwargs['date']
        if 'feature' in kwargs:
            if kwargs['feature'] is None or len(kwargs['feature']) == 0 or np.array_equal(self.feature , kwargs['feature']):
                pass
            else:
                self.on_change_feature()
                self.feature = kwargs['feature']
        return self.asserted()

    def asserted(self):
        if self.secid is not None:
            self.secid = self.as_array(self.secid)
        if self.date is not None:
            self.date = self.as_array(self.date)
        if self.feature is not None:
            self.feature = self.as_array(self.feature)

        if self.values is not None: 
            if isinstance(self.values , (int , float)):
                self.values = torch.full((len(self.secid),len(self.date),1,len(self.feature)),self.values)
            if not isinstance(self.values , torch.Tensor):
                self.values = torch.Tensor(self.values)
            if self.ndim == 3: 
                self.values = self.values.unsqueeze(2)

        if self.initiated:
            assert isinstance(self.values , torch.Tensor) , self.values
            assert self.ndim == 4 , self.shape
            assert self.shape[0] == len(self.secid) , (self.shape[0] , len(self.secid))
            assert self.shape[1] == len(self.date) , (self.shape[1] , len(self.date))
            assert self.shape[3] == len(self.feature) , (self.shape[3] , len(self.feature))
        return self
    
    def __repr__(self):
        if self.initiated:
            return f'{self.__class__.__name__}(values={self.shape},secid={str(self.secid)},date={str(self.date)},feature={self.feature})'
        else:
            return f'{self.__class__.__name__}()'

    def __len__(self):
        return len(self.values) if self.initiated else 0

    @property
    def initiated(self): 
        return self.values is not None
    @property
    def shape(self): 
        return properties.shape(self.values)
    @property
    def dtype(self): 
        return None if self.values is None else self.values.dtype
    @property
    def ndim(self): 
        return None if self.values is None else self.values.ndim
    @property
    def empty(self): 
        return not self.initiated or properties.empty(self.values)
    @property
    def max_date(self): 
        return properties.max_of_date(self.date)
    @property
    def min_date(self): 
        return properties.min_of_date(self.date)
    @property
    def inday(self) -> np.ndarray: 
        return np.arange(self.shape[2])
    @property
    def first_valid_date(self):
        dates = self.valid_dates
        return dates[0] if len(dates) > 0 else 99991231

    @property
    def last_valid_date(self):
        dates = self.valid_dates
        return dates[-1] if len(dates) > 0 else 19000101

    @property
    def valid_dates(self):
        if self.empty:
            return np.array([],dtype = int)
        return self.date[self.values.isfinite().any(dim = (0,2,3)).cpu().detach().numpy()] if self.initiated else np.array([],dtype = int)

    def set_flags(self , **kwargs):
        if not self.initiated:
            # will not set flags if the block is not initiated
            return self
        if not hasattr(self , '_flags'):
            self._flags = {}
        self._flags.update(kwargs)
        return self

    def check_flags(self , **kwargs):
        for key , value in kwargs.items():
            if self.flags[key] != value:
                raise ValueError(f'Invalid flags: {self.flags} , compare to {kwargs} try set then first before checking!')
        return self

    @property
    def flags(self):
        if not hasattr(self , '_flags'):
            self._flags = {}
        return self._flags

    def clear_flags(self):
        if hasattr(self , '_flags'):
            self._flags.clear()
        return self

    def on_change_feature(self):
        ...
    
    def date_within(self , start : int | None = None , end : int | None = None , interval = 1) -> np.ndarray:
        date = self.date
        if start is not None:
            date = date[date >= start]
        if end is not None:
            date = date[date <= end]
        return date[::interval]
    
    @classmethod
    def merge(cls , block_list , * , inplace = False , 
              secid_method : Literal['intersect' , 'union' , 'stack' , 'check'] = 'union' , 
              date_method : Literal['intersect' , 'union' , 'stack' , 'check'] = 'union' , 
              inday_method : Literal['intersect' , 'union' , 'stack' , 'check'] = 'check' , 
              feature_method : Literal['intersect' , 'union' , 'stack' , 'check'] = 'stack'):
        """merge multiple blocks into one block , if inplace is True, merge into the first block"""
        blocks = [blk for blk in block_list if isinstance(blk , cls) and not blk.empty]
        if len(blocks) == 0: 
            return cls()
        elif len(blocks) == 1: 
            return blocks[0] if inplace else blocks[0].copy()
            
        secid   = index_merge([blk.secid   for blk in blocks] , method = secid_method)
        date    = index_merge([blk.date    for blk in blocks] , method = date_method)
        inday   = index_merge([blk.inday   for blk in blocks] , method = inday_method)
        feature = index_merge([blk.feature for blk in blocks] , method = feature_method)

        values = torch.full((len(secid),len(date),len(inday),len(feature)) , torch.nan)
        
        for i , blk in enumerate(blocks): 
            tar_grid , src_grid = intersect_meshgrid([secid , date , inday , feature] , [blk.secid , blk.date , blk.inday , blk.feature] , )
            values[*tar_grid] = blk.values[*src_grid].to(values)

        block = blocks[0] if inplace else blocks[0].copy() 
        block.update(values = values , secid = secid , date = date , feature = feature)
        return block

    def merge_others(self , others : list | Any , inplace = False):
        if not isinstance(others , list): 
            others = [others]
        self = self.merge([self , *others] , inplace = inplace)
        self = self.align_feature(self.feature , inplace = True)
        return self
    
    def to(self , *args , **kwargs):
        if not self.initiated:
            return self
        self.values = self.values.to(*args , **kwargs)
        return self
    
    def copy(self): 
        return deepcopy(self)

    def align(self , secid = None , date = None , feature = None , inplace = False):
        if not self.initiated:
            return self
        blk = self.align_secid_date(secid , date , inplace = inplace)
        blk = blk.align_feature(feature , inplace = True)
        return blk

    def subset(self , secid : Any | None = None , date : Any | None = None , feature : Any | None = None , inday : Any | None = None , fillna : Any = None):
        if not self.initiated:  
            return self
        values  = self.loc(secid , date , feature , inday , fillna)
        secid = self.secid if secid is None else secid
        date = self.date if date is None else date
        feature = self.feature if feature is None else feature
        return self.__class__(values , secid , date , feature)

    def align_secid(self , secid , inplace = False):
        if not self.initiated:
            return self
        secid = None if secid is None else self.as_array(secid)
        if not inplace:
            self = self.copy()
        if secid is None or len(secid) == 0 or np.array_equal(secid , self.secid): 
            return self
        elif np.isin(secid , self.secid).all():
            return self.update(values = self.loc(secid = secid) , secid = secid)
        values = torch.full((len(secid) , *self.shape[1:]) , np.nan).to(self.values)
        tar_pos , src_pos = intersect_pos_slice(secid , self.secid)
        values[tar_pos] = self.values[src_pos]
        return self.update(values = values , secid = secid)
       
    def align_date(self , date , inplace = False):
        if not self.initiated:
            return self
        date = None if date is None else self.as_array(date)
        if not inplace:
            self = self.copy()
        if date is None or len(date) == 0 or np.array_equal(date , self.date): 
            return self
        elif np.isin(date , self.date).all():
            return self.update(values = self.loc(date = date) , date = date)
        values = torch.full((self.shape[0] , len(date) , *self.shape[2:]) , np.nan).to(self.values)
        tar_pos , src_pos = intersect_pos_slice(date , self.date)
        values[:,tar_pos] = self.values[:,src_pos]
        return self.update(values = values , date = date)

    def slice_date(self , start : int | None = None , end : int | None = None):
        if not self.initiated:
            return self
        start = start or self.date[0]
        end = end or self.date[-1]
        if start > self.date[0] or end < self.date[-1]:
            return self.align_date(self.date_within(start , end) , inplace = True)
        else:
            return self
    
    def align_secid_date(self , secid = None , date = None , inplace = False):
        # to speed up than .align_secid(secid = secid).align_date(date = date)
        if not self.initiated:
            return self
        secid = None if secid is None else self.as_array(secid)
        date = None if date is None else self.as_array(date)
        if not inplace:
            self = self.copy()
        if (secid is None or len(secid) == 0) and (date is None or len(date) == 0): 
            return self
        elif secid is None or len(secid) == 0 or np.array_equal(secid , self.secid):
            return self.align_date(date = date, inplace = True)
        elif date is None or len(date) == 0 or np.array_equal(date , self.date):
            return self.align_secid(secid = secid, inplace = True)
        elif np.isin(secid , self.secid).all() or np.isin(date , self.date).all():
            return self.align_date(date = date, inplace = True).align_secid(secid = secid, inplace = True)
        else:
            values = torch.full((len(secid),len(date),*self.shape[2:]) , np.nan).to(self.values)
            tar_grid , src_grid = intersect_meshgrid([secid , date] , [self.secid , self.date])
            values[*tar_grid] = self.values[*src_grid]
            
            return self.update(values = values , secid = secid , date = date)
    
    def align_feature(self , feature , inplace = False):
        if not self.initiated:
            return self
        feature = None if feature is None else self.as_array(feature) 
        if not inplace:
            self = self.copy()
        if feature is None or len(feature) == 0 or np.array_equal(feature , self.feature): 
            return self
        if np.isin(feature , self.feature).all():
            return self.update(values = self.loc(feature = feature) , feature = feature)
        values = torch.full((*self.shape[:-1],len(feature)) , np.nan).to(self.values)
        tar_pos , src_pos = intersect_pos_slice(feature , self.feature)
        values[...,tar_pos] = self.values[...,src_pos]
        self = self.update(values = values , feature = feature)
        self.on_change_feature()
        return self
        
    def add_feature(self , new_feature , new_value : np.ndarray | torch.Tensor):
        if not self.initiated:
            return self
        assert new_value.shape == self.shape[:-1] , (new_value.shape , self.shape[:-1])
        new_value = new_value.reshape(*new_value.shape , 1)
        self.values  = torch.concatenate([self.values,torch.Tensor(new_value)],dim=-1)
        self.feature = np.concatenate([self.feature,[new_feature]],axis=0)
        self.on_change_feature()
        return self
    
    def rename_feature(self , rename_dict : dict):
        if not self.initiated or len(rename_dict) == 0: 
            return self
        feature = self.feature.astype(object)
        for k,v in rename_dict.items(): 
            feature[feature == k] = v
        self.feature = feature.astype(str)
        self.on_change_feature()
        return self
    
    def loc(self , secid : Any | None = None , date : Any | None = None , feature : Any | None = None , inday : Any | None = None , fillna : Any = None):
        values = self.values
        if values is None:
            return values

        if feature is not None: 
            values = values[...,match_slice(feature , self.feature)]

        if inday is not None: 
            values = values[:,:,match_slice(inday , self.inday)]

        if date is not None: 
            values = values[:,match_slice(date , self.date)]

        if secid is not None: 
            values = values[match_slice(secid , self.secid)]

        if fillna is not None: 
            values = values.nan_to_num(fillna)
        return values

    @classmethod
    def concat_feature(cls , block_list):
        blocks = [blk for blk in block_list if isinstance(blk , cls) and blk.initiated] 
        for i , blk in enumerate(blocks): 
            if i == 0:
                new_blk = blk.copy()
            else:
                assert np.array_equal(new_blk.secid , blk.secid) , (new_blk.secid , blk.secid)
                assert np.array_equal(new_blk.date , blk.date) , (new_blk.date , blk.date)
                new_blk.feature = np.concatenate([new_blk.feature , blk.feature])
                new_blk.values  = torch.concatenate([new_blk.values  , blk.values] , dim=-1)
        new_blk.on_change_feature()
        return new_blk

    @classmethod
    def from_polars(cls , df : pl.DataFrame | None):
        """convert polars dataframe to stock4d"""
        if df is None or df.is_empty(): 
            return cls()

        # 1. Define the unique keys (keep these eager for the shape)
        assert 'secid' in df.columns and 'date' in df.columns , f'{df.columns} must contain secid and date'
        inday_marks = [inday_mark for inday_mark in INDAY_MARK_COLUMNS if inday_mark in df.columns]
        assert len(inday_marks) <= 1 , f'{df.columns} must contain less than one of {INDAY_MARK_COLUMNS}'
        secid = df['secid'].unique().sort()
        date  = df['date'].unique().sort()
        if inday_marks:
            if inday_marks[0] != 'inday':
                df = df.with_columns(pl.col(inday_marks[0]).cast(pl.Int64).alias('inday'))
            inday = df['inday'].unique().sort()
        else:
            df = df.with_columns(pl.lit(0).alias('inday'))
            inday = pl.Series('inday', [0])
        feature = [c for c in df.columns if c not in ['secid','date','inday']]

        # 2. Use the Lazy API for the heavy lifting
        # This creates a "Plan" that Polars will optimize before running
        grid_lazy = secid.to_frame().join(date.to_frame(), how="cross").join(inday.to_frame(), how="cross").lazy()
        values = (
            grid_lazy
            .join(df.lazy(), on=['secid', 'date' , 'inday'], how="left")
            .sort(['secid', 'date' , 'inday']) 
            .select(feature)
            .collect()
        ).to_numpy()
        values = torch.from_numpy(values).reshape(secid.len(), date.len(), inday.len(), len(feature))
        block = cls(values , secid.to_numpy() , date.to_numpy() , feature)
        return block

    
    @classmethod
    def from_pandas(cls , df : pd.DataFrame | None):
        if df is None or df.empty: 
            return cls()
        try:
            df = df.reset_index().drop(columns = ['index'] , errors = 'ignore').set_index(['secid' , 'date'])
            inday_marks = [inday_mark for inday_mark in INDAY_MARK_COLUMNS if inday_mark in df.columns]
            assert len(inday_marks) <= 1 , f'{df.columns} must contain less than one of {INDAY_MARK_COLUMNS}'
            if inday_marks:
                df = df.rename(columns = {inday_marks[0]:'inday'}).set_index('inday' , append = True)
            xarr = NdData.from_xarray(xr.Dataset.from_dataframe(df))
        except Exception as e:
            Logger.error(f'Failed to convert DataFrame to NdData: {e}')
            Logger.print_exc(e)
            Logger.display(df[df.index.duplicated()] , caption = 'Duplicate index in DataFrame')
            raise
        try:
            block = cls(xarr.values , xarr.index[0] , xarr.index[1] , xarr.index[-1])
        except Exception:
            import src
            setattr(src , 'xarr' , xarr)
            Logger.stdout(xarr)
            raise
        return block

    def to_dataframe(self , drop_inday = True , start : int | None = None , end : int | None = None):
        if start is not None or end is not None:
            date_slice = np.repeat(True,len(self.date))
            if start is not None: 
                date_slice[self.date < start] = False
            if end   is not None: 
                date_slice[self.date > end]   = False
            values = self.values[:,date_slice]
            date = self.date[date_slice]
        else:
            values = self.values
            date = self.date
        if drop_inday and values.shape[2] == 1:
            df_index = pd.MultiIndex.from_product([self.secid.tolist() , date.tolist()] , names = ['secid' , 'date'])
        else:
            df_index = pd.MultiIndex.from_product([self.secid.tolist() , date.tolist() , self.inday.tolist()] , names = ['secid' , 'date' , 'inday'])
        df = pd.DataFrame(values.flatten(end_dim=-2).cpu().numpy() , index = df_index , columns = self.feature)
        return df