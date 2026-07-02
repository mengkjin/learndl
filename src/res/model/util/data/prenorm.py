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
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from src.proj import Logger , Base
from src.data import DataBlockNorm 
from src.res.model.util.config import ModelConfig

__all__ = ['PrenormOperator']

@dataclass(slots=True)
class SingleDataPrenorm:
    name : str
    divlast  : bool = False
    histnorm : bool = False
    channelnorm : bool = False
            
    def __bool__(self):
        return self.divlast or self.histnorm or self.channelnorm

    def __call__(self , x : torch.Tensor , histnorm : DataBlockNorm | None = None , features : Base.alias.NamesType = None) -> torch.Tensor:
        return self.prenorm(x , histnorm , features)

    def prenorm(self , x : torch.Tensor , histnorm : DataBlockNorm | None = None , features : Base.alias.NamesType = None) -> torch.Tensor:
        """
        return panel-normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        3.channelnorm: normalized by channel avg
        """
        if hasattr(SpecialDataPrenorm, f'sp_{self.name}'):
            return getattr(SpecialDataPrenorm, f'sp_{self.name}')(x , features)
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

class SpecialDataPrenorm:
    @classmethod
    def sp_30mcont(cls , x : torch.Tensor , features : Base.alias.NamesType = None) -> torch.Tensor:
        assert features is not None , 'features is required for 30mcont'
        features = Base.ensure_name_list(features , [])
        assert 'open' in features , 'open is required for 30mcont'
        assert x.ndim == 4 , f'x must be 4-dimensional for 30mcont, but got {x.shape}'
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        price_idx = torch.tensor([idx for idx , feat in enumerate(features) if feat in ('close' , 'high' , 'low' , 'open' , 'vwap')])
        open_idx = torch.tensor([idx for idx , feat in enumerate(features) if feat == 'open'])
        x[...,price_idx] = x[...,price_idx] / (x[...,open_idx][:,:1] + 1e-6)
        return x

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

    def prenorm(self , key : str , x : torch.Tensor , features : Base.alias.NamesType = None) -> torch.Tensor:
        """
        return panel_normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        """
        return self[key](x , self.histnorms.get(key , None) , features)
