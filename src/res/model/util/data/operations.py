"""
Special operations for data module datas.
Includes:
- PrenormOperator: prenorm operator for data module datas
  - prenorm data according to the prenorm method : divlast, histnorm, channelnorm
- DataOperator: operations for data module datas
  - standardize y and weight
  - rolling rotation
  - finite position
  - split sample
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from numpy.random import permutation
from torch.utils.data import BatchSampler

from src.proj import Logger , Base
from src.func.tensor import nanmedian , standardize , rank_pct
from src.data import DataBlockNorm 
from src.data.preprocess import PrePros
from src.res.model.util.config import ModelConfig

from .batch_input_loader import DataloaderParam

__all__ = ['PrenormOperator' , 'DataOperator']

@dataclass(slots=True)
class SingleDataPrenorm:
    name : str
    divlast  : bool = False
    histnorm : bool = False
    channelnorm : bool = False
            
    def __bool__(self):
        return self.divlast or self.histnorm or self.channelnorm

    def __call__(self , x : torch.Tensor , histnorm : DataBlockNorm | None = None) -> torch.Tensor:
        return self.prenorm(x , histnorm)

    def prenorm(self , x : torch.Tensor , histnorm : DataBlockNorm | None = None) -> torch.Tensor:
        """
        return panel-normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        3.channelnorm: normalized by channel avg
        """
        assert not ((self.histnorm and histnorm is not None) and (self.channelnorm and x.ndim > 2)) , f'histnorm and channelnorm cannot be used together'
        if self.divlast and x.shape[-2] > 1:
            x = x / (x.select(-2,-1).unsqueeze(-2) + 1e-6)
        if self.histnorm and histnorm is not None:
            x = x - histnorm.avg[-x.shape[-2]:]
            x = x / (histnorm.std[-x.shape[-2]:] + 1e-6)
            if self.divlast and x.shape[-2] > 1:
                x[:,-1] = 0
        if self.channelnorm and x.ndim > 2:
            norm_dim = tuple(range(1 , x.ndim - 1))
            x = x / (x.mean(dim = norm_dim, keepdim = True).abs() + 1e-6) - 1
        return x

    @classmethod
    def from_input(cls, name : str , prenorm_method: dict[str, Any] | None = None):
        prenorm_method = prenorm_method or {}
        return cls(
            name = name ,
            divlast = prenorm_method.get('divlast'  , False) and (name in DataBlockNorm.DIVLAST) ,
            histnorm = prenorm_method.get('histnorm' , False) and (name in DataBlockNorm.HISTNORM) ,
            channelnorm = prenorm_method.get('channelnorm' , False)
        )

class PrenormOperator:
    """prenorm operator for data module datas"""
    def __init__(self, config: ModelConfig , histnorms : dict[str, DataBlockNorm] | None = None):
        self.config = config
        self.histnorms = histnorms or {}
        
        self.prenorms = {name: SingleDataPrenorm.from_input(name, self.config.input_data_prenorm.get(name)) for name in self.input_keys}
        [Logger.success(f'{name} {prenorm} initialized' , vb_level = 'max') for name , prenorm in self.prenorms.items() if prenorm]

    def __repr__(self):
        return f'{self.__class__.__name__}(prenorms={self.prenorms})'

    def __getitem__(self , key : str) -> SingleDataPrenorm:
        return self.prenorms.get(key , self.empty_prenormer)

    @cached_property
    def empty_prenormer(self) -> SingleDataPrenorm:
        return SingleDataPrenorm(name = 'empty')

    @property
    def input_keys(self) -> list[str]:
        return self.config.input_data_types

    def prenorm(self , key : str , x : torch.Tensor) -> torch.Tensor:
        """
        return panel_normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        """
        return self[key](x , self.histnorms.get(key , None))

class DataOperator:
    """operations for data module datas"""
    def __init__(self, config: ModelConfig , loader_param: DataloaderParam | None = None):
        self.config = config
        self.loader_param = loader_param or DataloaderParam()

    def __repr__(self):
        return f'{self.__class__.__name__}(config={self.config})'

    def update_loader_param(self , loader_param: DataloaderParam):
        self.loader_param = loader_param

    @property
    def stage(self) -> Base.lit.StageAll:
        return self.loader_param.stage

    @property
    def seq_lens(self) -> dict[str,int]:
        return self.loader_param.seqlens

    @property
    def seq_steps(self) -> dict[str,int]:
        return self.config.seq_steps

    def get_seqlen_step(self , key : str | None) -> tuple[int,int]:
        """get seqlen and step for a given key"""
        if not key:
            return 1 , 1
        else:
            assert key in self.seq_lens and key in self.seq_steps , (key , self.seq_lens , self.seq_steps)
            return self.seq_lens[key] , self.seq_steps[key]

    @staticmethod
    def _index1_date_window(
        index1 : torch.Tensor , window : int , date_len : int
    ) -> tuple[int , int , torch.Tensor]:
        """Return ``[start, end)`` slice bounds and ``index1`` relative to ``start`` for rolling checks."""
        idx_min = int(index1.min().item())
        idx_max = int(index1.max().item())
        start = max(0 , idx_min - window + 1)
        end = min(date_len , idx_max + 1)
        return start , end , index1 - start

    def standardize_y(
        self , y : torch.Tensor , effective : torch.Tensor | None = None , index1 : torch.Tensor | None = None , no_weight = False) -> tuple[torch.Tensor , torch.Tensor | None]:
        """standardize y and weight"""
        y = y[:,index1].clone() if index1 is not None else y.clone()
        if effective is not None: 
            y.nan_to_num_(0)[~effective] = torch.nan
        y = standardize(y , dim=0)
        weight_scheme = self.config.weight_scheme(self.stage , no_weight)
        w = self.label_weighting(weight_scheme , y)
        return y, w

    def label_weighting(self , weight_scheme : Base.lit.ConfigWeightScheme , y : torch.Tensor) -> torch.Tensor | None:
        """weighting for label"""
        if weight_scheme == 'equal' or y.isnan().all().item(): 
            return None
        if weight_scheme == 'top':
            w = torch.ones_like(y)
            try: 
                w[y > nanmedian(y , dim=0 , keepdim=True)] = 2
            except Exception:    
                w[y > nanmedian(y)] = 2
        elif weight_scheme == 'polar':
            y_rank = rank_pct(y , dim = 0)
            w = torch.abs(2 * y_rank - 1).square().clamp(0.25 , 1)
        else:
            raise ValueError(f'Invalid weight scheme: {weight_scheme}')
        return w

    def rolling_rotation(
        self , key : str | None , x : torch.Tensor , 
        index0 : torch.Tensor | Any , index1 : torch.Tensor | Any , * , 
        dim = 1 , squeeze_out = True
    ) -> torch.Tensor:
        """rotate [stock , date , inday , feature] to [sample , rolling sequence (by step) , inday , feature]"""
        
        seqlen , step = self.get_seqlen_step(key)
        assert x.ndim == 4 , x.ndim
        assert len(index0) == len(index1) , (index0 , index1)
        assert index1.max() < x.shape[dim] , (index1.max() , x.shape)
        assert index1.min() >= seqlen * step - 1 , (index1.min() , seqlen , step)
        
        try:
            start = max(0 , index1.min().item() - seqlen * step + 1)
            end = min(x.shape[dim] , index1.max().item() + 1)
            new_index1 = index1 - start + 1 - seqlen * step
            new_x = x[:,start:end].unfold(dim , seqlen * step , 1)[index0 , new_index1].\
                permute(0,3,1,2)[:,step-1::step] # [stock , seqlen (by step) , inday , feature]
        except MemoryError:
            new_x = torch.stack([x[index0 , index1 + (i + 1 - seqlen) * step] for i in range(seqlen)],dim=dim)
        
        assert new_x.shape[1] == seqlen , (new_x.shape[1] , seqlen)
        if squeeze_out: 
            new_x = new_x.squeeze(-2)
        return new_x

    def effective_samples(self , x : dict[str,torch.Tensor] , y : torch.Tensor | None , index1 : torch.Tensor ) -> torch.Tensor | None:
        """
        return effective sample position (with shape of len(index[0]) * step_len) the first 2 dims
        effective sample:
            1. x should be non-nan, all for nn, any for others
            2. x should be active, that is for any channel, block of seq_len * inday should be chaning any
        x : rolling window (seqlen * step) non-nan , end non-zero if in k is divlast
        y : endpoint non-nan if y is not None
        """
        effs : list[torch.Tensor] = []
        if x:
            finites = torch.stack([self.finite_position(k , v , index1) for k , v in x.items()] , dim = -1)
            eff = finites.all(dim=-1) if self.config.module_type == 'nn' else finites.any(dim=-1)
            effs.append(eff)
        if x and self.config.input_data_types:
            keys = [k for k in x.keys() if k in self.config.input_data_types and not PrePros.allow_inactive(k)]
            if keys:
                eff = torch.stack([self.active_position(k , x[k] , index1) for k in keys] , dim = -1).any(dim=-1)
                effs.append(eff)
        if y is not None:
            eff = self.finite_position(None , y, index1)
            effs.append(eff)
        if effs:
            return torch.stack(effs , dim = -1).all(dim = -1)
        else:
            return None

    def finite_position(self , key : str | None , data : torch.Tensor , index1 : torch.Tensor) -> torch.Tensor:
        """return finite position (with shape of len(index[0]) * step_len) the first 2 dims"""
        require_all = self.config.module_type == 'nn'
        endpoint_nonzero = key and (key in DataBlockNorm.DIVLAST)
        seqlen , step = self.get_seqlen_step(key)
        assert data.ndim > 2 , data.ndim
        window = seqlen * step
        sum_dim = tuple(range(2,data.ndim))
        if window == 1:
            start , end , rel_index1 = self._index1_date_window(index1 , 1 , data.shape[1])
            data_slice = data[:,start:end]
            if require_all:
                fin = data_slice.sum(sum_dim).isfinite()
            else:
                fin = ~data_slice.isnan().all(sum_dim)
            finite = fin[:,rel_index1]
        else:
            start , end , rel_index1 = self._index1_date_window(index1 , window , data.shape[1])
            data_slice = data[:,start:end]
            if require_all:
                fin = data_slice.sum(sum_dim).isfinite()
            else:
                fin = ~data_slice.isnan().all(sum_dim)
            pad_fin = torch.nn.functional.pad(fin, (window - 1,0) , value = False)
            try:
                finite = pad_fin.unfold(1,window,1)[...,step-1::step][:,rel_index1]
                finite = finite.all(dim=-1) if require_all else finite.any(dim=-1)
            except MemoryError:
                abs_index1 = index1
                if require_all:
                    finite = torch.full((len(data),len(index1)), True).to(dtype = torch.bool , device = data.device)
                    for i in range(seqlen):
                        finite &= pad_fin[:,abs_index1 - start + (i + 1) * step - 1]
                else:
                    finite = torch.full((len(data),len(index1)), False).to(dtype = torch.bool , device = data.device)
                    for i in range(seqlen):
                        finite |= pad_fin[:,abs_index1 - start + (i + 1) * step - 1]
        if endpoint_nonzero:
            finite &= data[:,index1].not_equal(0).all(sum_dim)
        return finite

    def active_position(self , key : str | None , data : torch.Tensor , index1 : torch.Tensor) -> torch.Tensor:
        """return active position (with shape of len(index[0]) * step_len) the first 2 dims"""
        seqlen , step = self.get_seqlen_step(key)
        assert data.ndim > 2 , data.ndim
        window = seqlen * step
        if window <= 2:
            return torch.full((len(data),len(index1)) , True).to(dtype = torch.bool , device = data.device)

        start , end , rel_index1 = self._index1_date_window(index1 , window , data.shape[1])
        data_slice = data[:,start:end]
        reduce_dim = tuple(range(2 , data_slice.ndim - 1))
        if data_slice.ndim == 3:
            avg = data_slice
            pos_std = torch.full(avg.shape[:2] , False).to(dtype = torch.bool , device = data.device)
        else:
            avg = data_slice.mean(dim = reduce_dim)
            pos_std = (data_slice.std(dim = reduce_dim , unbiased = False) > 0).any(-1)

        pad_avg = torch.nn.functional.pad(avg, (0 , 0 , window - 1 , 0) , mode = 'replicate')
        pad_std = torch.nn.functional.pad(pos_std, (window - 1 , 0) , value = 0)
        try:
            windows = pad_avg.unfold(1 , window , 1)[...,step-1::step][:,rel_index1]
            moving_avg = self._rolling_feature_activity(windows)
            moving_std = (pad_std.unfold(1 , window , 1)[...,step-1::step][:,rel_index1] > 0).any(-1)
            active = moving_avg | moving_std
        except MemoryError:
            active = torch.full((len(data),len(index1)), False).to(dtype = torch.bool , device = data.device)
            avg_benchmark = pad_avg[:,rel_index1 + step - 1]
            for i in range(seqlen):
                active |= (pad_avg[:,rel_index1 + (i + 1) * step - 1] - avg_benchmark).to(torch.bool).any(-1)
                active |= pad_std[:,rel_index1 + (i + 1) * step - 1] > 0
        return active.nan_to_num_(False)

    @staticmethod
    def _rolling_feature_activity(windows : torch.Tensor) -> torch.Tensor:
        """Whether any feature shows cross-sectional dispersion at any step in a rolling window."""
        feat_dim = windows.shape[-1]
        if feat_dim <= 1:
            return (windows.std(dim = -1 , unbiased = False) > 0).any(-1)
        chunk = 32
        active = torch.zeros(windows.shape[:-2] , dtype = torch.bool , device = windows.device)
        for f0 in range(0 , feat_dim , chunk):
            sub = windows[..., f0:f0 + chunk]
            active |= (sub.std(dim = -1 , unbiased = False) > 0).any(-1)
        return active

    def split_sample(self , effective : torch.Tensor , index0 : torch.Tensor , index1 : torch.Tensor) -> dict[str,list[torch.Tensor]]:
        """
        update index of train/valid sub-samples of flattened all-samples(with in 0:len(index[0]) * step_len - 1)
        sample_tensor should be boolean tensor , True indicates non

        train/valid sample method: total_shuffle , sequential , both_shuffle , train_shuffle
        test sample method: sequential
        """
        sample_method = self.config.sample_method
        train_ratio = self.config.train_ratio
        batch_size = self.config.batch_size
        l0 , l1 = effective.shape[:2]
        pos = torch.stack([index0.repeat_interleave(l1) , index1.repeat(l0)] , -1).reshape(l0,l1,2)
        
        def _shuffle(i , bs = batch_size):
            return [i[p] for p in BatchSampler(permutation(np.arange(len(i))) , bs , drop_last=False)]
        def _sequential(beg , end , posit = pos , effective = effective):
            return [posit[:,j][effective[:,j]] for j in range(beg , end)]

        sample_index = {}
        if self.stage == 'fit':
            sep = int(l1 * train_ratio)
            assert sep > 0 and sep < l1 , (sep , l1 , train_ratio)
            if sample_method == 'total_shuffle':
                pool = torch.Tensor(permutation(np.arange(effective.sum().item())))
                sep = int(len(pool) * train_ratio)
                assert sep > 0 and sep < len(pool) , (sep , len(pool) , train_ratio)
                sample_index['train'] = _shuffle(pos[effective][pool[:sep]])
                sample_index['valid'] = _shuffle(pos[effective][pool[sep:]])
            elif sample_method == 'both_shuffle':
                sample_index['train'] = _shuffle(pos[:,:sep][effective[:,:sep]])
                sample_index['valid'] = _shuffle(pos[:,sep:][effective[:,sep:]])
            elif sample_method == 'train_shuffle':
                sample_index['train'] = _shuffle(pos[:,:sep][effective[:,:sep]])
                sample_index['valid'] = _sequential(sep , l1)
            else:
                sample_index['train'] = _sequential(0 , sep)
                sample_index['valid'] = _sequential(sep , l1)
        else:
            sample_index[self.stage] = _sequential(0 , l1)

        return sample_index   