"""
DataBlock is a class that represents a block of stored data in tensor format (secid , date , inday , feature).
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr  

from copy import deepcopy

from dataclasses import dataclass
from pathlib import Path
from typing import Any , ClassVar , Literal , Iterable

from src.proj import PATH , Logger , CALENDAR , DB
from src.proj.util import properties , torch_load
from src.func import match_slice , forward_fillna , index_merge , intersect_meshgrid , intersect_pos_slice

from .nd import NdData
from ..stock_info import INFO

__all__ = ['DataBlock' , 'DataBlockNorm']

INDAY_MARK_COLUMNS = ('inday' , 'minute')
FREQUENT_DBS  = ('trade_ts.day' , 'trade_ts.day_val' , 'models.tushare_cne5_exp')
FREQUENT_MIN_DATES = 500
PREFERRED_DUMP_SUFFIXES = ('.mmap' , '.pt' , '.feather')


def data_type_abbr(key : str):
    """
    Normalise a data-type key to its canonical short form.

    Rules
    -----
    - ``trade_<suffix>`` → ``<suffix>``
    - ``rtn_lag*`` / ``res_lag*`` / ``std_lag*`` → ``<prefix><sum_of_lags>``
    - ``'y'`` or ``'labels'`` → ``'y'``
    - all other keys → unchanged (lowercased)
    """
    key = key.lower()
    if (key.startswith('trade_') and len(key)>6):
        return key[6:]
    elif key.startswith(('rtn_lag','res_lag','std_lag')):
        return '{:s}{:d}'.format(key[:3] , sum([int(s) for s in key[7:].split('_')]))
    elif key in ['y' , 'labels']:
        return 'y'
    else:
        return key

def data_type_alias(key : str) -> list[str]:
    """return possible alternatives for a key , the key itself must be the last one , so when iteration ends will use the input key"""
    alias = [f'trade_{key}' , key.replace('trade_','') , key]
    assert alias[-1] == key , f'{alias[-1]} != {key}'
    return alias

def save_dict(data : dict , file_path : str | Path):
    """
    Save a dictionary to disk.

    Supports ``.npz`` (numpy compressed) and ``.pt`` / ``.pth`` (torch pickle)
    formats, chosen by the file suffix.
    """
    file_path = Path(file_path)
    assert not file_path.exists() or file_path.is_file() , file_path
    Path(file_path).parent.mkdir(exist_ok=True)
    if file_path.suffix in ['.npz' , '.npy' , '.np']:
        np.savez_compressed(file_path , **data)
    elif file_path.suffix in ['.pt' , '.pth']:
        torch.save(data , file_path , pickle_protocol = 5)
    else:
        raise Exception(file_path)

def load_dict(file_path : str | Path , keys = None) -> dict[str,Any]:
    """
    Load a dictionary from disk.

    Supports ``.npz`` and ``.pt`` / ``.pth`` formats.  If ``keys`` is given,
    only those keys are returned (intersected with the file's actual keys).
    The file must exist; an ``AssertionError`` is raised otherwise.
    """
    file_path = Path(file_path)
    assert file_path.exists() and file_path.is_file() , file_path
    if file_path.suffix in ['.npz' , '.npy' , '.np']:
        file = np.load(file_path)
    elif file_path.suffix in ['.pt' , '.pth']:
        file = torch_load(file_path)
    else:
        raise Exception(file_path)
    keys = file.keys() if keys is None else np.intersect1d(keys , list(file.keys()))
    data = {k:file[k] for k in keys}
    return data

@dataclass
class DataBlock:
    """
    Core 4-D tensor container: ``(N_secid, N_date, N_inday, N_feature)``.

    The central data structure used throughout the model pipeline.  Each axis
    has a corresponding coordinate array:

    - ``secid``   : 1-D int array of stock identifiers
    - ``date``    : 1-D int array of trading dates (yyyyMMdd)
    - ``inday``   : implicit int range ``[0, N_inday)`` (1 for daily data)
    - ``feature`` : 1-D str array of feature names

    An uninitiated block (``values is None``) is treated as empty and acts as
    a neutral element in merge operations.

    Construction
    ------------
    - ``DataBlock(values, secid, date, feature)``   , direct
    - ``DataBlock.from_pandas(df)``                 , from long-format MultiIndex DataFrame
    - ``DataBlock.from_polars(df)``                 , from Polars DataFrame
    - ``DataBlock.load_raw(db_src, db_key, ...)``   , from database with optional caching
    - ``DataBlock.load_dump(...)``                  , from pre-saved ``.mmap`` / ``.pt`` / ``.feather``

    Key operations
    --------------
    - Alignment : ``align_secid``, ``align_date``, ``align_feature``, ``align_secid_date``
    - Merging   : ``merge(block_list, ...)``  — union/intersect/stack/check per axis
    - Slicing   : ``loc``, ``subset``, ``slice_date``
    - Transforms: ``adjust_price``, ``adjust_volume``, ``ffill``, ``mask_values``
    - Persistence: ``save_dump``, ``load_dump``, ``path_preprocess``
    - Normalisation: ``hist_norm``, ``load_preprocess_norms``
    """
    values  : torch.Tensor | Any = None 
    secid   : np.ndarray | Any = None 
    date    : np.ndarray | Any = None 
    feature : np.ndarray | Any = None

    def __post_init__(self) -> None:
        self.asserted()

    def uninitiate(self):
        """Reset all fields to None, producing an empty uninitiated block."""
        self.values  = None
        self.secid   = None
        self.date    = None
        self.feature = None

    @classmethod
    def as_array(cls , values : np.ndarray | torch.Tensor | list | tuple | str | int | float | Any) -> np.ndarray:
        """Convert various array-like inputs to a 1-D or N-D numpy array."""
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
        """
        Update one or more fields in-place and re-run ``asserted()``.

        Accepted keyword arguments: ``values``, ``secid``, ``date``, ``feature``.
        Changing ``feature`` calls ``on_change_feature()`` to invalidate any
        feature-dependent cached state.
        """
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
        """
        Validate and normalise all fields after construction or mutation.

        - Converts ``secid``, ``date``, ``feature`` to numpy arrays.
        - Ensures ``values`` is a 4-D ``torch.Tensor`` (auto-inserts inday=1 dim
          when a 3-D tensor is provided).
        - Asserts shape consistency between ``values`` and the coordinate arrays.
        """
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
        """Return a compact one-line summary of shape and index arrays."""
        if self.initiated:
            return f'{self.__class__.__name__}(values={self.shape},secid={str(self.secid)},date={str(self.date)},feature={self.feature})'
        else:
            return f'{self.__class__.__name__}()'

    def __len__(self):
        """Number of securities (first dimension), or 0 if uninitiated."""
        return len(self.values) if self.initiated else 0

    @property
    def initiated(self):
        """True if ``values`` is not None."""
        return self.values is not None

    @property
    def shape(self):
        """Shape of ``values`` as a tuple ``(N_secid, N_date, N_inday, N_feature)``."""
        return properties.shape(self.values)

    @property
    def dtype(self):
        """Torch dtype of ``values``, or None if uninitiated."""
        return None if self.values is None else self.values.dtype

    @property
    def ndim(self):
        """Number of dimensions (always 4 when initiated)."""
        return None if self.values is None else self.values.ndim

    @property
    def empty(self):
        """True if uninitiated or ``values`` has zero elements."""
        return not self.initiated or properties.empty(self.values)

    @property
    def max_date(self):
        """Maximum date in the ``date`` array, or None if empty."""
        return properties.max_of_date(self.date)

    @property
    def min_date(self):
        """Minimum date in the ``date`` array, or None if empty."""
        return properties.min_of_date(self.date)

    @property
    def inday(self) -> np.ndarray:
        """Integer range ``[0, N_inday)`` representing the intra-day bar indices."""
        return np.arange(self.shape[2])

    @property
    def first_valid_date(self):
        """First date that has at least one finite value; returns 99991231 if none."""
        dates = self.valid_dates
        return dates[0] if len(dates) > 0 else 99991231

    @property
    def last_valid_date(self):
        """Last date that has at least one finite value; returns 19000101 if none."""
        dates = self.valid_dates
        return dates[-1] if len(dates) > 0 else 19000101

    @property
    def valid_dates(self):
        """Array of dates for which at least one finite value exists across secid/inday/feature."""
        if self.empty:
            return np.array([],dtype = int)
        return self.date[self.values.isfinite().any(dim = (0,2,3)).cpu().detach().numpy()] if self.initiated else np.array([],dtype = int)

    def set_flags(self , **kwargs):
        """
        Attach arbitrary metadata flags to this block (e.g. ``category``, ``db_src``).
        No-op if the block is uninitiated.  Returns ``self`` for chaining.
        """
        if not self.initiated:
            # will not set flags if the block is not initiated
            return self
        if not hasattr(self , '_flags'):
            self._flags = {}
        self._flags.update(kwargs)
        return self

    def check_flags(self , **kwargs):
        """Assert that the current flags match the given key=value pairs."""
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
        """Remove all metadata flags from this block."""
        if hasattr(self , '_flags'):
            self._flags.clear()
        return self
    
    def date_within(self , start : int | None = None , end : int | None = None , interval = 1) -> np.ndarray:
        """Return the subset of ``self.date`` within ``[start, end]``, optionally strided."""
        date = self.date
        if start is not None:
            date = date[date >= start]
        if end is not None:
            date = date[date <= end]
        return date[::interval]
    
    @classmethod
    def merge(cls , block_list : Iterable[DataBlock] , * , inplace = False , 
              secid_method : Literal['intersect' , 'union' , 'stack' , 'check'] = 'union' , 
              date_method : Literal['intersect' , 'union' , 'stack' , 'check'] = 'union' , 
              inday_method : Literal['intersect' , 'union' , 'stack' , 'check'] = 'check' , 
              feature_method : Literal['intersect' , 'union' , 'stack' , 'check'] = 'stack'):
        """merge multiple blocks into one block , if inplace is True, merge into the first block"""
        blocks = [*block_list]
        if inplace:
            assert len(blocks) >= 1 , 'merge: inplace is True, but block_list is empty'
            target_block = blocks[0]
        else:
            target_block = cls()
        merge_blocks = [blk for blk in blocks if isinstance(blk , cls) and not blk.empty]
        if len(merge_blocks) == 0 or (len(merge_blocks) == 1 and (merge_blocks[0] is target_block)): 
            return target_block
            
        secid   = index_merge([blk.secid   for blk in merge_blocks] , method = secid_method)
        date    = index_merge([blk.date    for blk in merge_blocks] , method = date_method)
        inday   = index_merge([blk.inday   for blk in merge_blocks] , method = inday_method)
        feature = index_merge([blk.feature for blk in merge_blocks] , method = feature_method)

        values = torch.full((len(secid),len(date),len(inday),len(feature)) , torch.nan)
        
        for i , blk in enumerate(merge_blocks): 
            tar_grid , src_grid = intersect_meshgrid([secid , date , inday , feature] , [blk.secid , blk.date , blk.inday , blk.feature] , )
            values[*tar_grid] = blk.values[*src_grid].to(values)

        target_block.update(values = values , secid = secid , date = date , feature = feature)
        return target_block

    def merge_others(self , *others : DataBlock , inplace = False):
        """Merge one or more additional blocks into this block; equivalent to ``merge([self, *others])``."""
        self = self.merge([self , *others] , inplace = inplace)
        self = self.align_feature(self.feature , inplace = True)
        return self
    
    def to(self , *args , **kwargs):
        """Cast ``values`` to a different dtype or device (forwarded to ``torch.Tensor.to``)."""
        if not self.initiated:
            return self
        self.values = self.values.to(*args , **kwargs)
        return self
    
    def copy(self):
        """Return a deep copy of this block."""
        return deepcopy(self)

    def align(self , secid = None , date = None , feature = None , inplace = False):
        """Convenience wrapper: align secid+date then feature in one call."""
        if not self.initiated:
            return self
        blk = self.align_secid_date(secid , date , inplace = inplace)
        blk = blk.align_feature(feature , inplace = True)
        return blk

    def subset(self , secid : Any | None = None , date : Any | None = None , feature : Any | None = None , inday : Any | None = None , fillna : Any = None):
        """
        Return a new DataBlock containing only the requested index slices.

        Unlike ``align_*`` methods, ``subset`` always returns a new block
        (never in-place) and does not fill missing positions with NaN.
        """
        if not self.initiated:  
            return self
        values  = self.loc(secid , date , feature , inday , fillna)
        secid = self.secid if secid is None else secid
        date = self.date if date is None else date
        feature = self.feature if feature is None else feature
        return self.__class__(values , secid , date , feature)

    def align_secid(self , secid , inplace = False):
        """
        Re-index the security axis to ``secid``.

        Missing secids are filled with NaN; extra secids in the current block
        are dropped.  If ``secid`` is a sub-set of ``self.secid``, a fast
        index-lookup path is used instead of creating a new full tensor.
        """
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
        """
        Re-index the date axis to ``date``.

        Missing dates are filled with NaN; extra dates in the current block
        are dropped.  Same fast-path optimisation as ``align_secid``.
        """
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
        """Restrict the date axis to [start, end] without creating NaN-padded rows."""
        if not self.initiated:
            return self
        start = start or self.date[0]
        end = end or self.date[-1]
        if start > self.date[0] or end < self.date[-1]:
            return self.align_date(self.date_within(start , end) , inplace = True)
        else:
            return self
    
    def align_secid_date(self , secid = None , date = None , inplace = False):
        """
        Jointly re-index both secid and date axes in a single tensor allocation.

        Faster than chaining ``align_secid`` + ``align_date`` when both axes
        need to change, because it avoids an intermediate tensor.
        """
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
        """
        Re-index the feature axis to ``feature``.

        Missing features are filled with NaN; extra features in the current
        block are dropped.  Calls ``on_change_feature()`` when the feature
        set changes.
        """
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
        """Append a new feature column (and its values) to the block in-place."""
        if not self.initiated:
            return self
        assert new_value.shape == self.shape[:-1] , (new_value.shape , self.shape[:-1])
        new_value = new_value.reshape(*new_value.shape , 1)
        self.values  = torch.concatenate([self.values,torch.Tensor(new_value)],dim=-1)
        self.feature = np.concatenate([self.feature,[new_feature]],axis=0)
        self.on_change_feature()
        return self
    
    def rename_feature(self , rename_dict : dict):
        """Rename features according to ``rename_dict`` (old_name → new_name)."""
        if not self.initiated or len(rename_dict) == 0: 
            return self
        feature = self.feature.astype(object)
        for k,v in rename_dict.items(): 
            feature[feature == k] = v
        self.feature = feature.astype(str)
        self.on_change_feature()
        return self
    
    def loc(self , secid : Any | None = None , date : Any | None = None , feature : Any | None = None , inday : Any | None = None , fillna : Any = None):
        """
        Extract a sub-tensor by positional index matching (not alignment).

        Only elements present in the given index arrays are returned; the
        output shape reflects the intersection sizes, not the requested sizes.
        ``fillna`` replaces NaN in the output when provided.
        """
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
        """
        Concatenate blocks along the feature axis.

        All blocks must share identical ``secid`` and ``date`` arrays.
        Returns a new block with features from all input blocks stacked in order.
        """
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
        """convert polars dataframe to DataBlock"""
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
        """convert pandas dataframe to DataBlock"""
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
        """
        Convert the block to a long-format pandas DataFrame.

        Index is ``MultiIndex [secid, date]`` (or ``[secid, date, inday]`` when
        ``drop_inday=False`` and N_inday > 1).  Columns match ``self.feature``.
        Optionally restricts to dates in ``[start, end]``.
        """
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


    @property
    def price_adjusted(self): 
        """Return flag of if the price is adjusted by adjfactor"""
        if not hasattr(self , '_price_adjusted'):
            self._price_adjusted = False
        return self._price_adjusted

    @price_adjusted.setter
    def price_adjusted(self , value : bool):
        self._price_adjusted = value

    @property
    def volume_adjusted(self): 
        """Return flag of if the volume is adjusted by adjfactor"""
        if not hasattr(self , '_volume_adjusted'):
            self._volume_adjusted = False
        return self._volume_adjusted

    @volume_adjusted.setter
    def volume_adjusted(self , value : bool):
        self._volume_adjusted = value

    @staticmethod
    def data_type_abbr(key : str): 
        """Return the abbreviation of the data type"""
        return data_type_abbr(key)

    @staticmethod
    def data_type_alias(key : str): 
        """Return the alias of the data type"""
        return data_type_alias(key)

    @classmethod
    def last_preprocess_date(cls , key , type : Literal['fit' , 'predict']):
        """Return the calendar date (yyyyMMdd int) when the preprocess dump was last written."""
        path = cls.path_preprocess(key , type)
        if path.suffix == '.mmap':
            dates = [PATH.file_modified_date(sub_path) for sub_path in path.iterdir() if sub_path.is_file()] if path.exists() else []
            return min(dates) if dates else None
        else:
            return PATH.file_modified_date(cls.path_preprocess(key , type))
    
    @classmethod
    def last_preprocess_time(cls , key , type : Literal['fit' , 'predict']):
        """Return the wall-clock timestamp (datetime) when the preprocess dump was last written."""
        path = cls.path_preprocess(key , type)
        if path.suffix == '.mmap':
            times = [PATH.file_modified_time(sub_path) for sub_path in path.iterdir() if sub_path.is_file()] if path.exists() else []
            return min(times) if times else None
        else:
            return PATH.file_modified_time(path)

    @classmethod
    def last_data_date(cls , key : str , type : Literal['fit' , 'predict']):
        """Return the maximum date stored in the preprocess dump for ``key`` / ``type``."""
        try:
            path = cls.path_preprocess(key , type)
            if not path.exists():
                return None
            if path.suffix == '.mmap':
                return max(load_dict(path.joinpath('index.pt'))['date'])
            elif path.suffix == '.pt':
                return max(load_dict(path)['date'])
            elif path.suffix == '.feather':
                return max(pd.read_feather(path)['date'])
            else:
                raise ValueError(f'Unsupported suffix: {path.suffix}')
        except ModuleNotFoundError as e:
            Logger.error(f'last_data_date({key , type}) error: ModuleNotFoundError: {e}')
            return None

    def ffill(self , if_fill : bool = True):
        """Forward-fill NaN values along the date axis (axis=1). No-op when ``if_fill=False``."""
        if self.empty:
            return self
        if if_fill:
            self.values = forward_fillna(self.values , axis = 1)
        return self

    def fillna(self , value : Any = 0):
        """Replace all NaN values in ``values`` with the given scalar."""
        if self.empty:
            return self
        if isinstance(self.values , torch.Tensor):
            self.values = self.values.nan_to_num(value)
        elif isinstance(self.values , np.ndarray):
            self.values = np.nan_to_num(self.values , value)
        else:
            raise TypeError(f'Unsupported type: {type(self.values)} for {self.__class__.__name__} values')
        return self
        
    @staticmethod
    def guess_fillna(name : str , fillna : Literal['guess'] | bool | None = 'guess' ,
                     excl : tuple[str,...] = ('y','day','15m','min','30m','60m','week')) -> bool:
        """
        Decide whether a given block type should be forward-filled.

        When ``fillna='guess'``, returns True for any block whose key does
        *not* start with the excluded prefixes (i.e. factor blocks are
        filled, while return/OHLCV blocks are left raw).
        """
        if fillna == 'guess':
            return name.startswith(excl) == 0
        else:
            return bool(fillna)

    def on_change_feature(self):
        """
        Hook called whenever the feature axis is modified.

        Clears the ``'raw'`` category flag so that a block whose features
        have been transformed is not accidentally re-saved as raw data.
        """
        if self.flags.get('category') == 'raw':
            self.clear_flags()
        return self

    def adjust_price(self , adjfactor = True , multiply : Any = 1 , divide : Any = 1 ,
                     price_feat = ['preclose' , 'close', 'high', 'low', 'open', 'vwap']):
        """
        Apply price adjustment factors to OHLCV price columns.

        If ``adjfactor=True`` and an ``adjfactor`` feature is present, prices are
        multiplied by it.  Optional scalar ``multiply`` and ``divide`` factors are
        applied afterwards.  Sets ``price_adjusted=True`` to prevent double-adjustment.
        NaN vwap values are back-filled from the close (or nearest available price).
        """
        if self.price_adjusted or self.empty: 
            return self
        adjfactor = adjfactor and ('adjfactor' in self.feature)
        if multiply is None and divide is None and (not adjfactor): 
            return self  

        if isinstance(price_feat , (str,)): 
            price_feat = [price_feat]
        i_price = np.where(np.isin(self.feature , price_feat))[0].astype(int)
        if len(i_price) == 0: 
            return self
        v_price = self.values[...,i_price]

        if adjfactor :  
            v_price *= self.loc(feature=['adjfactor'] , fillna = 1)
        if multiply  is not None: 
            v_price *= multiply
        if divide    is not None: 
            v_price /= divide
        self.values[...,i_price] = v_price 

        if 'vwap' in self.feature:
            i_vp = np.where(self.feature == 'vwap')[0].astype(int)
            nan_idx = self.values[...,i_vp].isnan() if isinstance(self.values , torch.Tensor) else np.isnan(self.values[...,i_vp])
            nan_idx = nan_idx.squeeze(-1)
            pcols = [col for col in ['close', 'high', 'low', 'open' , 'preclose'] if col in self.feature]
            if pcols: 
                i_cp = np.where(self.feature == pcols[0])[0].astype(int)
                self.values[nan_idx , i_vp] = self.values[nan_idx , i_cp]
            else:
                ...
        
        self.price_adjusted = True
        return self
    
    def adjust_volume(self , multiply = None , divide = None ,
                      vol_feat = ['volume' , 'amount', 'turn_tt', 'turn_fl', 'turn_fr']):
        """
        Scale volume/amount/turnover columns by optional multiply/divide factors.

        Sets ``volume_adjusted=True`` to prevent double-adjustment.
        No-op if both ``multiply`` and ``divide`` are None.
        """
        if self.volume_adjusted or self.empty: 
            return self
        if multiply is None and divide is None: 
            return self

        if isinstance(vol_feat , (str,)): 
            vol_feat = [vol_feat]
        i_vol = np.where(np.isin(self.feature , vol_feat))[0]
        if len(i_vol) == 0: 
            return self
        v_vol = self.values[...,i_vol]
        if multiply is not None: 
            v_vol *= multiply
        if divide   is not None: 
            v_vol /= divide
        self.values[...,i_vol] = v_vol
        self.volume_adjusted = True
        return self
    
    def mask_values(self , mask : dict , **kwargs):
        """
        Zero out (set to NaN) values according to mask rules.

        Currently supports ``mask = {'list_dt': N_days}`` which blanks out
        values for dates before each stock's listing date plus an offset.
        Additional mask types can be added as keys in the ``mask`` dict.
        """
        if not mask or self.empty: 
            return self
        mask_pos = torch.full_like(self.values , fill_value=False , dtype=torch.bool)
        if mask_list_dt := mask.get('list_dt'):
            desc = INFO.get_desc(set_index=False)
            desc = desc[desc['secid'] > 0].loc[:,['secid','list_dt','delist_dt']]
            if len(np.setdiff1d(self.secid , desc['secid'])) > 0:
                add_df = pd.DataFrame({
                        'secid':np.setdiff1d(self.secid , desc['secid']) ,
                        'list_dt':21991231 , 'delist_dt':21991231})
                desc = pd.concat([desc,add_df],axis=0).reset_index(drop=True)

            desc = desc.sort_values('list_dt',ascending=False).drop_duplicates(subset=['secid'],keep='first').set_index('secid') 
            secid , date = self.secid , self.date
            
            list_dt = np.array(desc.loc[secid , 'list_dt'])
            list_dt[list_dt < 0] = 21991231
            list_dt = CALENDAR.cd_array(list_dt , mask_list_dt).astype(int)

            delist_dt = np.array(desc.loc[secid , 'delist_dt'])
            delist_dt[delist_dt < 0] = 21991231

            tmp = torch.from_numpy(np.stack([(date <= lst) + (date >= dls) for lst,dls in zip(list_dt , delist_dt)] , axis = 0))
            mask_pos[tmp] = True

        assert (~mask_pos).sum() > 0 , 'all values are masked'
        self.values[mask_pos] = torch.nan
        return self
    
    def hist_norm(self , key : str , 
                  start : int | None = None , end : int | None  = 20161231 , 
                  step_day = 5 , **kwargs):
        """Calculate the historical normalisation stats for the data block"""
        return DataBlockNorm.calculate(self , key , start , end , step_day , **kwargs)

    def extend_to(self , db_src : str , db_key : str , start : int | None = None , end : int | None = None , * ,
                  dates = None , feature : list[str] | None = None , use_alt = True , inplace = True , vb_level : Any = 'max'):
        """
        Extend this block by loading missing dates from the database.

        Only dates in ``dates`` (or ``start``–``end``) that are absent from
        ``self.date`` are fetched.  Price/volume adjustment flags are forwarded
        to the newly loaded block before merging.
        """
        if dates is None:
            dates = CALENDAR.range(start , end , 'td')
        block = self.load_raw(db_src , db_key , dates = CALENDAR.diffs(dates , self.date) , feature = feature , use_alt = use_alt , vb_level = vb_level)
        if self.price_adjusted:
            block = block.adjust_price()
        if self.volume_adjusted:
            block = block.adjust_volume()
        self = self.merge_others(block , inplace = inplace)
        return self

    @classmethod
    def path_preprocess(cls , key : str , type : Literal['fit' , 'predict'] , * ,
                        dump_suffix : Literal['.mmap' , '.pt' , '.feather'] = '.mmap' , find_if_not_exists = True) -> Path:
        """
        Return the filesystem path for the preprocessed dump of ``key`` / ``type``.

        Falls back to the first existing suffix in ``PREFERRED_DUMP_SUFFIXES``
        when ``find_if_not_exists=True`` and the canonical path does not exist.
        """
        if key.lower() in ['y' , 'labels']: 
            path = PATH.block.joinpath(type , f'Y{dump_suffix}')
        else:
            alias_list = data_type_alias(key)
            for new_key in alias_list:
                path = PATH.block.joinpath(type , f'X_{new_key}{dump_suffix}')
                if path.exists(): 
                    break
        if find_if_not_exists:
            return cls.find_existing_dump_path(path)
        return path
        
    @staticmethod
    def path_norm(key : str , type : Literal['fit'] = 'fit'):
        """Return the path to the normalisation stats for ``key`` / ``type``."""
        return DataBlockNorm.norm_path(key , type)

    @classmethod
    def path_raw(cls , src : str , key : str , * , dump_suffix : Literal['.mmap' , '.pt' , '.feather'] = '.mmap' , find_if_not_exists = True):
        """Return the path to the raw data for ``src`` / ``key``."""
        raw_path = PATH.block.joinpath('raw' , f'{src}.{key}{dump_suffix}')
        if find_if_not_exists:
            return cls.find_existing_dump_path(raw_path)
        return raw_path

    @classmethod
    def find_existing_dump_path(cls , raw_path : Path) -> Path:
        """Find the existing dump path for the raw data"""
        if raw_path.exists():
            return raw_path
        for suffix in PREFERRED_DUMP_SUFFIXES:
            new_path = raw_path.with_suffix(suffix)
            if new_path.exists():
                return new_path
        return raw_path

    @classmethod
    def load_preprocess(cls , key : str , type : Literal['fit' , 'predict'] , **kwargs) -> DataBlock:
        """Load a preprocessed block; for ``type='predict'`` extends the date range to today."""
        block = cls.load_dump(category = 'preprocess' , type = type , preprocess_key = key)
        if type == 'predict' and key == 'y' and not block.empty:
            block = block.align_date(CALENDAR.range(min(block.date) , CALENDAR.updated() , 'td'))
        return block

    @classmethod
    def blocks_align(cls , blocks : dict[str,DataBlock] , * , start = None , end = None ,
                     intersect_secid = True , inplace : Literal[True] = True , vb_level : Any = 2) -> dict[str,DataBlock]:
        """
        Align a dict of blocks to a common (secid, date) grid.

        secid: intersection of all blocks (when ``intersect_secid=True``).
        date : union of all blocks, clipped to ``[start, end]`` and trimmed
               so no block starts before its own first available date.
        All operations are in-place by default.
        """
        if len(blocks) <= 1:
            return blocks
        
        block_title = f'{len(blocks)} DataBlocks' if len(blocks) > 3 else f'DataBlock [{",".join(blocks.keys())}]'
        with Logger.Timer(f'Align {block_title}' , vb_level = vb_level):
            # sligtly faster than .align(secid = secid , date = date)
            if intersect_secid:  
                newsecid = index_merge([blk.secid for blk in blocks.values()] , method = 'intersect')
                
            else:
                newsecid = None
            
            newdate = index_merge([blk.date for blk in blocks.values()] , method = 'union' , min_value = start , max_value = end)
            max_min_date = max([min(blk.date) for blk in blocks.values() if not blk.empty])
            newdate = newdate[newdate >= max_min_date]
            
            for blk in blocks.values():
                blk.align_secid_date(newsecid , newdate , inplace = inplace)

        return blocks

    @classmethod
    def blocks_ffill(cls , blocks : dict[str,DataBlock] , * ,
                      fillna : Literal['guess'] | bool | None = 'guess' , exclude : Iterable[str] | None = None) -> dict[str,DataBlock]:
        """Apply forward-fill to each block in the dict, optionally excluding specific keys."""
        exclude = exclude or []
        fillnas = {key:cls.guess_fillna(key , fillna) for key in blocks}
        for key , blk in blocks.items():
            if key in exclude:
                continue
            blk.ffill(fillnas[key])
        return blocks

    @classmethod
    def load_preprocess_norms(cls , keys : list[str] | str , type : Literal['fit'] = 'fit' , dtype = None) -> dict[str,DataBlockNorm]:
        """Load the normalisation stats for the data block"""
        if isinstance(keys , str):
            keys = [keys]
        return DataBlockNorm.load_keys(keys, type , dtype = dtype)

    @classmethod
    def load_from_db(cls , db_src : str , db_key : str , start = None , end = None , * , 
                     dates = None , feature = None , use_alt = True , vb_level : Any = 'max') -> DataBlock:
        """Load the data block from the database"""
        return cls.load_from_db_polars(db_src , db_key , start , end , dates = dates , feature = feature , use_alt = use_alt , vb_level = vb_level)

    @classmethod
    def load_from_db_pandas(
        cls , db_src : str , db_key : str , start = None , end = None , * , 
        dates = None , feature = None , use_alt = True , vb_level : Any = 'max'
    ) -> DataBlock:
        """Load the data block from the database using pandas dataframe , usually slower than polars"""
        if dates is None:
            dates = CALENDAR.range(start , end , 'td')

        df = DB.loads(db_src , db_key , dates = dates , use_alt=use_alt , fill_datavendor=True , vb_level=vb_level)
        block = cls.from_pandas(df) if len(df) > 0 else cls()
        if feature is None:
            block.set_flags(category = 'raw' , db_src = db_src , db_key = db_key)
        return block

    @classmethod
    def load_from_db_polars(
        cls , db_src : str , db_key : str , start = None , end = None , * , 
        dates = None , feature = None , use_alt = True , vb_level : Any = 'max'
    ):
        """Load the data block from the database using polars dataframe , usually faster than pandas"""
        if dates is None:
            dates = CALENDAR.range(start , end , 'td')
        df = DB.loads_pl(db_src , db_key , dates = dates , use_alt=use_alt , fill_datavendor=True , vb_level=vb_level)
        block = cls.from_polars(df) if df.height > 0 else cls()
        if feature is None:
            block.set_flags(category = 'raw' , db_src = db_src , db_key = db_key)
        return block
        
    @classmethod
    def load_raw(cls , db_src : str , db_key : str , start = None , end = None , * ,
                 dates = None , feature = None , use_alt = True , vb_level : Any = 'max'):
        """
        Load a block from the database, with smart caching for frequent DB keys.

        For DB keys listed in ``FREQUENT_DBS``, a local ``.mmap`` dump is
        maintained and only missing dates are fetched from the database.
        All other keys are loaded directly on each call.
        """
        if dates is None:
            dates = CALENDAR.range(start , end , 'td')

        if f'{db_src}.{db_key}' in FREQUENT_DBS:
            if len(dates) >= FREQUENT_MIN_DATES:
                block = cls.load_dump(category = 'raw' , db_src = db_src , db_key = db_key)
                loaded = True
            else:
                block = cls()
                loaded = False
            saved_dates = block.date if block.date is not None else np.array([])
            update_dates = CALENDAR.diffs(dates , saved_dates)
            if len(update_dates) > 0:
                new_block = cls.load_from_db(db_src , db_key , dates = update_dates , use_alt = use_alt , vb_level = vb_level) # no feature selection here
                block = block.merge_others(new_block , inplace = True)
            if (len(update_dates) > 0 and loaded) or not cls.path_raw(db_src , db_key).exists():
                block.save_dump()
            block = block.align(date = dates , feature = feature , inplace = True)
        else:
            block = cls.load_from_db(db_src , db_key , dates = dates , feature = feature , use_alt = use_alt , vb_level = vb_level)
        return block

    @classmethod
    def load_dump(cls , **kwargs) -> DataBlock:
        """
        Load a block from a pre-saved dump file.

        The ``flags`` kwargs determine which path to read from:
        - ``category='preprocess'``: reads from the preprocessed dump path
        - ``category='raw'``: reads from the raw cached dump path

        Supports ``.mmap`` (memory-mapped), ``.pt`` (torch pickle), and
        ``.feather`` formats.  Returns an empty block if no file is found.
        """
        flags = kwargs
        if flags.get('category') == 'preprocess':
            path = cls.path_preprocess(flags['preprocess_key'] , flags['type'])
        else:
            path = cls.path_raw(flags['db_src'] , flags['db_key'])
        
        if not path.exists():
            for suffix in PREFERRED_DUMP_SUFFIXES:
                path = path.with_suffix(suffix)
                if path.exists():
                    break

        if path.exists():
            if path.suffix == '.mmap':
                assert path.is_dir() , path
                values = DB.ArrayMemoryMap.load_tensor(path.joinpath('values'))
                index = load_dict(path.joinpath('index.pt'))
                block = cls(values , index['secid'] , index['date'] , index['feature'])
            elif path.suffix == '.pt':
                block = cls(**load_dict(path))
            elif path.suffix == '.feather':
                block = cls.from_pandas(pd.read_feather(path))
            else:
                raise ValueError(f'Unsupported suffix: {path.suffix}')
        else:
            block = cls()
        return block.set_flags(**flags)

    def save_dump(self):
        """
        save the block to PATH.block
        """
        if self.empty:
            return
        flags = self.flags
        if flags.get('category') == 'raw':
            assert not self.price_adjusted and not self.volume_adjusted , f'price and volume must not be adjusted before saving!'
            assert f'{flags["db_src"]}.{flags["db_key"]}' in FREQUENT_DBS , f'{flags["db_src"]}.{flags["db_key"]} is not a frequent db!'
            path = self.path_raw(flags['db_src'] , flags['db_key'])
        elif flags.get('category') == 'preprocess':
            path = self.path_preprocess(flags['preprocess_key'] , flags['type'])
        else:
            raise ValueError(f'Unsupported category: {flags.get("category")} , please set correct category before saving!')
        path.parent.mkdir(exist_ok=True)
        if path.suffix == '.feather':
            assert not path.exists() or path.is_file() , path
            df = self.to_dataframe()
            df.to_feather(path) 
        elif path.suffix == '.mmap':
            assert not path.exists() or path.is_dir() , path
            path.mkdir(parents=True, exist_ok=True)
            DB.ArrayMemoryMap.save(self.values , path.joinpath('values'))
            save_dict({'date' : self.date , 'secid' : self.secid , 'feature' : self.feature} , path.joinpath('index.pt'))
        elif path.suffix == '.pt':
            assert not path.exists() or path.is_file() , path
            save_dict({'values' : self.values , 'date' : self.date.astype(int) , 'secid' : self.secid.astype(int) , 'feature' : self.feature} , path)
        else:
            raise ValueError(f'Unsupported suffix: {path.suffix}')

    @classmethod
    def fix_dumps(cls):
        """Fix the dump of the data block to the preferred dump suffix"""
        category_path = PATH.block.joinpath('raw')
        category = 'raw'
    
        for path in category_path.iterdir():
            with Logger.Timer(f'Change {category}.{path.name} dump method'):
                new_path = path.with_suffix(PREFERRED_DUMP_SUFFIXES[0])
                db_src , db_key = path.name.split('.')[:2]
                block = cls.load_from_db(db_src , db_key , 20070101 , 20241231)
                block.save_dump()
                Logger.success(f'{category}.{path.name} changed to {new_path}')

@dataclass(slots=True)
class DataBlockNorm:
    """
    Historical normalisation statistics for a single DataBlock data type.

    Stores ``avg`` (mean) and ``std`` (standard deviation) tensors of shape
    ``(N_inday * maxday, N_feature)`` computed by :meth:`DataBlock.hist_norm`.
    During model training the block values are divided by the endpoint value
    (for ``'day'``-type data) and then standardised using these statistics.

    Class Attributes
    ----------------
    DIVLAST : list[str]
        Data types whose values are divided by the last bar before normalising.
    HISTNORM : list[str]
        Data types for which historical normalisation is applied at all.
    """
    avg : torch.Tensor
    std : torch.Tensor
    dtype : Any = None

    # calculation method for histnorm, do not change for training. Instead, change the prenorm method in configs/model/input.yaml
    DIVLAST  : ClassVar[list[str]] = ['day']
    HISTNORM : ClassVar[list[str]] = ['day','15m','min','30m','60m']

    def __post_init__(self):
        """Cast avg and std to self.dtype after construction."""
        self.avg = self.avg.to(self.dtype)
        self.std = self.std.to(self.dtype)

    @classmethod
    def calculate(cls , block : DataBlock , key : str ,
                  start : int | None = None , end : int | None  = 20161231 ,
                  step_day = 5 , **kwargs):
        """
        Compute and persist historical normalisation statistics for a DataBlock.

        Samples the block at ``step_day`` intervals over the ``[start, end]``
        date range, building rolling windows of ``maxday`` days.  For ``DIVLAST``
        types, values are divided by the window endpoint before computing
        mean and std.  Saves the result to the norm path and returns it.

        Returns None for data types not in ``HISTNORM``.
        """
        
        key = data_type_abbr(key)
        if (key not in cls.HISTNORM): 
            return None

        default_maxday = {'day' : 60 , 'week' : 60}
        maxday = default_maxday.get(key , 1)

        date_slice = np.repeat(True , len(block.date))
        if start is not None: 
            date_slice[block.date < start] = False
        if end   is not None: 
            date_slice[block.date > end]   = False

        secid , date , inday , feat = block.secid , block.date , block.shape[2] , block.feature

        len_step = len(date[date_slice]) // step_day
        len_bars = maxday * inday

        x = torch.Tensor(block.values[:,date_slice])
        pad_array = (0,0,0,0,maxday,0,0,0)
        x = torch.nn.functional.pad(x , pad_array , value = torch.nan)
        
        avg_x , std_x = torch.zeros(len_bars , len(feat)) , torch.zeros(len_bars , len(feat))

        x_endpoint = x.shape[1]-1 + step_day * np.arange(-len_step + 1 , 1)
        x_div = torch.ones(len(secid) , len_step , 1 , len(feat)).to(x)
        re_shape = (*x_div.shape[:2] , -1)
        if key in cls.DIVLAST: # divide by endpoint , day dataset only
            x_div.copy_(x[:,x_endpoint,-1:])
            
        nan_sample = (x_div == 0).reshape(*re_shape).any(dim = -1)
        nan_sample += x_div.isnan().reshape(*re_shape).any(dim = -1)
        for i in range(maxday):
            nan_sample += x[:,x_endpoint-i].reshape(*re_shape).isnan().any(dim=-1)

        for i in range(maxday):
            vijs = ((x[:,x_endpoint - maxday+1 + i]) / (x_div + 1e-6))[nan_sample == 0]
            avg_x[i*inday:(i+1)*inday] = vijs.mean(dim = 0)
            std_x[i*inday:(i+1)*inday] = vijs.std(dim = 0)

        assert avg_x.isnan().sum() + std_x.isnan().sum() == 0 , ((nan_sample == 0).sum())
        
        data = cls(avg_x , std_x)
        data.save(key)
        return data

    def save(self , key):
        """Save avg and std tensors to the norm path for ``key``."""
        path = self.norm_path(key)
        path.parent.mkdir(exist_ok=True)
        save_dict({'avg' : self.avg , 'std' : self.std} , self.norm_path(key))

    @classmethod
    def load_keys(cls , keys : str | list[str] , type : Literal['fit'] , dtype = None) -> dict[str,DataBlockNorm]:
        """Load normalisation stats for multiple keys from disk; skips missing keys silently."""
        if not isinstance(keys , list): 
            keys = [keys]
        norms = {}
        for key in keys:
            path = cls.norm_path(key , type)
            if not path.exists(): 
                continue
            data = load_dict(path)
            norms[key] = cls(data['avg'] , data['std'] , dtype)
        return norms
    
    @classmethod
    def norm_path(cls , key : str , type : Literal['fit'] = 'fit'):
        """Return the path to the normalisation stats for ``key`` / ``type``."""
        if key.lower() == 'y':
            return PATH.norm.joinpath(type , 'Y.pt')
        alias_list = data_type_alias(key)
        for new_key in alias_list:
            path = PATH.norm.joinpath(type , f'X_{new_key}.pt')
            if path.exists():
                break
        return path