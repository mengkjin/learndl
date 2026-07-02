"""
DataBlockNorm is a class that represents the historical normalisation statistics for a DataBlock data frame.
"""

from __future__ import annotations

import torch
import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.proj import PATH , Base , Save , Load
from src.data.util import DataBlock

__all__ = ['PreProHistNorm']

HistNormTasks : tuple[str,...] = ('day' , 'week' , '30mcont' , 'min' , '15m' , '30m')
HistNorm : tuple[str,...] = ('day' , '30mcont')
DivLast : tuple[str,...] = ('day' , '30mcont')
SequenceLens : dict[str,int] = defaultdict(lambda: 1) | {'day': 60 , '30mcont': 30}
SequenceSteps : dict[str,int] = defaultdict(lambda: 1) 

@dataclass(slots=True)
class PreProHistNorm:
    """
    Historical normalisation statistics for a single PreProcessed DataBlock data frame.

    Stores ``avg`` (mean) and ``std`` (standard deviation) tensors of shape
    ``(N_inday * maxday, N_feature)`` computed by :meth:`DataBlock.hist_norm`.
    During model training the block values are divided by the endpoint value
    (for ``'day'``-frame data) and then standardised using these statistics.

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

    def __post_init__(self):
        """Cast avg and std to self.dtype after construction."""
        self.avg = self.avg.to(self.dtype)
        self.std = self.std.to(self.dtype)

    @classmethod
    def is_divlast(cls , key : str | None) -> bool:
        return key in DivLast
    
    @classmethod
    def has_histnorm(cls , key : str | None) -> bool:
        return key in HistNorm

    @classmethod
    def recalculate_all(cls) -> None:
        for key in HistNormTasks:
            block = DataBlock.load_preprocess(key , frame = 'fit')
            cls.calculate(block , key)

    @classmethod
    def calculate(
        cls , block : DataBlock , key : str ,
        start : int | None = None , end : int | None  = 20161231 ,
        calculate_interval = 5 , **kwargs
    ) -> PreProHistNorm | None:
        """
        Compute and persist historical normalisation statistics for a DataBlock.

        Samples the block at ``step_day`` intervals over the ``[start, end]``
        date range, building rolling windows of ``maxday`` days.  For ``DIVLAST``
        types, values are divided by the window endpoint before computing
        mean and std.  Saves the result to the norm path and returns it.

        Returns None for data types not in ``HISTNORM``.
        """
        
        key = DataBlock.data_type_abbr(key)
        if key not in HistNormTasks: 
            return None

        seq_len = SequenceLens.get(key , 1)
        seq_step = SequenceSteps.get(key , 1)

        date_slice = np.repeat(True , len(block.date))
        if start is not None: 
            date_slice[block.date < start] = False
        if end   is not None: 
            date_slice[block.date > end]   = False

        secid , date , inday , feat = block.secid , block.date , block.shape[2] , block.feature

        len_step = len(date[date_slice]) // calculate_interval
        len_bars = seq_len * inday

        x = torch.Tensor(block.values[:,date_slice])
        pad_array = (0,0,0,0,seq_len,0,0,0)
        x = torch.nn.functional.pad(x , pad_array , value = torch.nan)
        
        avg_x , std_x = torch.zeros(len_bars , len(feat)) , torch.zeros(len_bars , len(feat))

        x_endpoint = x.shape[1]-1 + calculate_interval * np.arange(-len_step + 1 , 1)
        x_div = torch.ones(len(secid) , len_step , 1 , len(feat)).to(x)
        re_shape = (*x_div.shape[:2] , -1)
        if cls.is_divlast(key): # divide by endpoint , day dataset only
            x_div.copy_(x[:,x_endpoint,-1:])
            if key == 'day':
                ...
            elif key == '30mcont':
                non_price_idx = torch.tensor(~np.isin(block.feature , ('close' , 'high' , 'low' , 'open' , 'vwap')))
                x_div[...,non_price_idx] = x[:,x_endpoint].sum(dim = -2, keepdim = True)[...,non_price_idx]
            else:
                raise ValueError(f'Unsupported divlast key: {key}')
        nan_sample = (x_div == 0).reshape(*re_shape).any(dim = -1)
        nan_sample += x_div.isnan().reshape(*re_shape).any(dim = -1)
        for i in range(seq_len):
            nan_sample += x[:,x_endpoint-i*seq_step].reshape(*re_shape).isnan().any(dim=-1)

        for i in range(seq_len):
            vijs = ((x[:,x_endpoint - (seq_len - 1 - i)*seq_step]) / (x_div + 1e-6))[nan_sample == 0]
            avg_x[i*inday:(i+1)*inday] = vijs.mean(dim = 0)
            std_x[i*inday:(i+1)*inday] = vijs.std(dim = 0)

        assert avg_x.isnan().sum() + std_x.isnan().sum() == 0 , ((nan_sample == 0).sum())
        
        data = cls(avg_x , std_x)
        data.save(key)
        return data

    def save(self , key : str) -> None:
        """Save avg and std tensors to the norm path for ``key``."""
        path = self.norm_path(key)
        path.parent.mkdir(exist_ok=True)
        Save.torch({'avg' : self.avg , 'std' : self.std} , self.norm_path(key))

    @classmethod
    def load_keys(cls , keys : Base.alias.NamesType , frame : Base.lit.DataBlockTimeFrame = 'fit' , dtype = None) -> dict[str,PreProHistNorm]:
        """Load normalisation stats for multiple keys from disk; skips missing keys silently."""
        assert frame == 'fit' , 'only fit frame is supported for normalisation stats'
        keys = Base.ensure_name_list(keys , [])
        norms : dict[str,PreProHistNorm] = {}
        for key in keys:
            path = cls.norm_path(key , frame)
            if not path.exists(): 
                continue
            data = Load.torch(path)
            norms[key] = cls(data['avg'] , data['std'] , dtype)
        return norms
    
    @classmethod
    def norm_path(cls , key : str , frame : Base.lit.DataBlockTimeFrame = 'fit') -> Path:
        """Return the path to the normalisation stats for ``key`` / ``frame``."""
        assert frame == 'fit' , 'only fit frame is supported for normalisation stats'
        if key.lower() == 'y':
            return PATH.histnorm.joinpath(frame , 'Y.pt')
        alias_list = DataBlock.data_type_alias(key)
        path = None
        for new_key in alias_list:
            path = PATH.histnorm.joinpath(frame , f'X_{new_key}.pt')
            if path.exists():
                break
        assert path , f'path not found for {key}'
        return path