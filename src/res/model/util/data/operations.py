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
from typing import Any , Callable , Literal
from numpy.random import permutation
from torch.utils.data import BatchSampler

from src.proj import Logger
from src.func.tensor import nanmedian , standardize , rank_pct
from src.data import DataBlockNorm
from src.res.model.util.config import ModelConfig

from .batch_input_loader import DataloaderParam

__all__ = ['PrenormOperator' , 'DataOperator']

@dataclass(slots=True)
class Prenormer:
    name : str
    divlast  : bool = False
    histnorm : bool = False
    channelnorm : bool = False
            
    def __bool__(self):
        return self.divlast or self.histnorm or self.channelnorm

    def __call__(self , x : torch.Tensor , histnorm : DataBlockNorm | Any = None) -> torch.Tensor:
        return self.prenorm(x , histnorm)

    def prenorm(self , x : torch.Tensor , histnorm : DataBlockNorm | Any = None) -> torch.Tensor:
        """
        return panel-normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        3.channelnorm: normalized by channel avg
        """
        option_divlast = self.divlast and x.shape[-2] > 1
        option_histnorm = self.histnorm and histnorm is not None
        option_channelnorm = self.channelnorm and x.ndim > 2
        assert not (option_histnorm and option_channelnorm) , f'histnorm and channelnorm cannot be used together'
        if option_divlast:
            x = x / (x.select(-2,-1).unsqueeze(-2) + 1e-6)
        if option_histnorm:
            x = x - histnorm.avg[-x.shape[-2]:]
            x = x / (histnorm.std[-x.shape[-2]:] + 1e-6)
            if option_divlast:
                x[:,-1] = 0
        if option_channelnorm:
            norm_dim = tuple(range(1 , x.ndim - 1))
            x = x / x.mean(dim = norm_dim, keepdim = True) + 1e-6 - 1
        return x

    @classmethod
    def from_input(cls, name : str , prenorm_method: dict[str, Any] | None):
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
        
        
        self.prenorms = {name: Prenormer.from_input(name, self.config.input_data_prenorm.get(name)) for name in self.input_keys}
        [Logger.success(f'{name} {prenorm} initialized' , vb_level = 'max') for name , prenorm in self.prenorms.items() if prenorm]

    def __repr__(self):
        return f'{self.__class__.__name__}(prenorms={self.prenorms})'

    def __getitem__(self , key : str) -> Prenormer:
        return self.prenorms.get(key , self.empty_prenormer)

    @cached_property
    def empty_prenormer(self) -> Prenormer:
        return Prenormer(name = 'empty')

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
    def __init__(self, config: ModelConfig , loader_param: DataloaderParam):
        self.config = config
        self.loader_param = loader_param

    @property
    def stage(self) -> Literal['fit' , 'test' , 'predict' , 'retrospective']:
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

    def standardize_y(
        self , y : torch.Tensor , valid : torch.Tensor | None = None , index1 : torch.Tensor | None = None , no_weight = False) -> tuple[torch.Tensor , torch.Tensor | None]:
        """standardize y and weight"""
        y = y[:,index1].clone() if index1 is not None else y.clone()
        if valid is not None: 
            y.nan_to_num_(0)[~valid] = torch.nan
        y = standardize(y , dim=0)
        weight_scheme = self.config.weight_scheme(self.stage , no_weight)
        w = self.label_weighting(weight_scheme , y)
        return y, w

    def label_weighting(self , weight_scheme : Literal['equal' , 'top' , 'polar'] , y : torch.Tensor) -> torch.Tensor | None:
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

    def rolling_rotation(self , key : str | None , x : torch.Tensor , index0 : torch.Tensor | Any , index1 : torch.Tensor | Any , * , dim = 1 , squeeze_out = True) -> torch.Tensor:
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

    def finite_position(self , key : str | None , data : torch.Tensor , index1 : torch.Tensor) -> torch.Tensor:
        """return finite position (with shape of len(index[0]) * step_len) the first 2 dims"""
        all_valid = self.config.module_type == 'nn'
        endpoint_nonzero = key and (key in DataBlockNorm.DIVLAST)
        seqlen , step = self.get_seqlen_step(key)
        assert data.ndim > 2 , data.ndim
        sum_dim = tuple(range(2,data.ndim))
        agg = data.sum(sum_dim).isfinite()
        if seqlen * step > 1:
            agg = torch.nn.functional.pad(agg, (seqlen * step - 1,0) , value = False)
        try:
            predicate : Callable[...,torch.Tensor] = torch.all if all_valid else torch.any
            valid = predicate(agg.unfold(1,seqlen*step,1)[...,step-1::step],-1)[:,index1]
        except MemoryError:
            predicate = torch.multiply if all_valid else torch.add
            valid = torch.full_like((agg[:,:len(index1)]), all_valid)
            for i in range(seqlen):
                valid = predicate(valid , agg[:,index1 + i * step])
        if endpoint_nonzero: 
            valid *= data[:,index1].not_equal(0).all(sum_dim)     
        return valid

    def split_sample(self , valid : torch.Tensor , index0 : torch.Tensor , index1 : torch.Tensor) -> dict[str,list[torch.Tensor]]:
        """
        update index of train/valid sub-samples of flattened all-samples(with in 0:len(index[0]) * step_len - 1)
        sample_tensor should be boolean tensor , True indicates non

        train/valid sample method: total_shuffle , sequential , both_shuffle , train_shuffle
        test sample method: sequential
        """
        sample_method = self.config.sample_method
        train_ratio = self.config.train_ratio
        batch_size = self.config.batch_size
        l0 , l1 = valid.shape[:2]
        pos = torch.stack([index0.repeat_interleave(l1) , index1.repeat(l0)] , -1).reshape(l0,l1,2)
        
        def _shuffle(i , bs = batch_size):
            return [i[p] for p in BatchSampler(permutation(np.arange(len(i))) , bs , drop_last=False)]
        def _sequential(beg , end , posit = pos , valid = valid):
            return [posit[:,j][valid[:,j]] for j in range(beg , end)]

        sample_index = {}
        if self.stage == 'fit':
            sep = int(l1 * train_ratio)
            if sample_method == 'total_shuffle':
                pool = torch.Tensor(permutation(np.arange(valid.sum().item())))
                sep = int(len(pool) * train_ratio)
                sample_index['train'] = _shuffle(pos[valid][pool[:sep]])
                sample_index['valid'] = _shuffle(pos[valid][pool[sep:]])
            elif sample_method == 'both_shuffle':
                sample_index['train'] = _shuffle(pos[:,:sep][valid[:,:sep]])
                sample_index['valid'] = _shuffle(pos[:,sep:][valid[:,sep:]])
            elif sample_method == 'train_shuffle':
                sample_index['train'] = _shuffle(pos[:,:sep][valid[:,:sep]])
                sample_index['valid'] = _sequential(sep , l1)
            else:
                sample_index['train'] = _sequential(0 , sep)
                sample_index['valid'] = _sequential(sep , l1)
        else:
            sample_index[self.stage] = _sequential(0 , l1)

        return sample_index   