import torch
from typing import Any
from src.proj import Logger
from src.data import DataBlockNorm
from src.res.model.util import ModelConfig

from dataclasses import dataclass

__all__ = ['PrenormOperator']

@dataclass(slots=True)
class Prenormer:
    name : str
    divlast  : bool = False
    histnorm : bool = False
    channelnorm : bool = False

    def __post_init__(self):
        if self: 
            Logger.success(f'Pre-Norm : {self}' , vb_level = 'max')
            
    def __bool__(self):
        return self.divlast or self.histnorm or self.channelnorm

    def prenorm(self , x : torch.Tensor , histnorm : DataBlockNorm | None = None) -> torch.Tensor:
        """
        return panel-normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        3.channelnorm: normalized by channel avg
        """
        if self.divlast and x.shape[-2] > 1:
            x = x / (x.select(-2,-1).unsqueeze(-2) + 1e-6)
        if self.histnorm and histnorm is not None:
            x = x - histnorm.avg[-x.shape[-2]:]
            x = x / (histnorm.std[-x.shape[-2]:] + 1e-6)
        if self.channelnorm and x.ndim > 2:
            x = x / (x.mean(dim = tuple(range(1 , x.ndim - 1)), keepdim = True) + 1e-6) - 1
        return x

    @classmethod
    def from_input(cls, name : str , prenorm_method: dict[str, Any] | None):
        prenorm_method = prenorm_method or {}
        return cls(
            name = name ,
            divlast = prenorm_method.get('divlast'  , False) and (name in DataBlockNorm.DIVLAST) ,
            histnorm = prenorm_method.get('histnorm' , False) and (name in DataBlockNorm.HISTNORM) ,
            channelnorm = prenorm_method.get('channelnorm' , True)
        )


class PrenormOperator:
    def __init__(self, config: ModelConfig , histnorms : dict[str, DataBlockNorm] | None = None):
        self.config = config
        self.histnorms = histnorms or {}
        self.prenorms = {name: Prenormer.from_input(name, self.config.input_data_prenorm.get(name)) for name in self.input_keys}

    @property
    def input_keys(self) -> list[str]:
        return self.config.input_data_types

    def prenorm(self , key : str , x : torch.Tensor) -> torch.Tensor:
        """
        return panel_normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        """
        return self.prenorms[key].prenorm(x , self.histnorms.get(key , None))