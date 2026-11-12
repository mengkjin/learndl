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
from src.data.preprocess import PreProHistNorm 
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

    def __call__(self , x : torch.Tensor , histnorm : PreProHistNorm | None = None , features : Base.alias.NamesType = None) -> torch.Tensor:
        return self.prenorm(x , histnorm , features)

    def prenorm(self , x : torch.Tensor , histnorm : PreProHistNorm | None = None , features : Base.alias.NamesType = None) -> torch.Tensor:
        """
        return panel-normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        3.channelnorm: normalized by channel avg
        """
        if self.histnorm:
            self.validate_histnorm(x, histnorm)
        else:
            histnorm = None
        if self.name == '30mcont':
            return SpecialDataPrenorm.sp_30mcont(x, histnorm, features, divlast=self.divlast)
        if hasattr(SpecialDataPrenorm, f'sp_{self.name}'):
            return getattr(SpecialDataPrenorm, f'sp_{self.name}')(x , histnorm = histnorm , features = features)
        assert not ((self.histnorm and histnorm is not None) and (self.channelnorm and x.ndim > 2)) , f'histnorm and channelnorm cannot be used together'
        if self.divlast and x.shape[-2] > 1:
            x = x / (x.select(-2,-1).unsqueeze(-2) + 1e-6)
        if histnorm is not None:
            if self.name == '15m':
                # [bars, features] broadcasts over both stocks and days.
                avg = histnorm.avg
                std = histnorm.std
            else:
                # Daily windows align to the endpoint of the historical window.
                window_length = x.shape[-2]
                avg = histnorm.avg[-window_length:]
                std = histnorm.std[-window_length:]
            x = (x - avg) / (std + 1e-6)
            if self.divlast and x.shape[-2] > 1:
                x[:, -1] = 0
        if self.channelnorm and x.ndim > 2:
            norm_dim = tuple(range(1 , x.ndim - 1))
            x = x / (x.mean(dim = norm_dim, keepdim = True).abs() + 1e-6) - 1
        return x

    def validate_histnorm(self, x: torch.Tensor, histnorm: PreProHistNorm | None) -> None:
        """Validate shared statistics, then check the dataset's time-axis contract."""
        if self.name in ('day', '30mcont') and not self.divlast:
            raise ValueError(f'{self.name}: histnorm requires divlast')
        if self.name == '15m':
            assert self.histnorm , '15m should use histnorm'
        if histnorm is None:
            raise ValueError(f'{self.name}: missing historical normalization statistics')

        avg, std = histnorm.avg, histnorm.std
        error = f'{self.name}: invalid historical normalization statistics'
        if avg.ndim != 2 or avg.shape != std.shape:
            raise ValueError(f'{error}: avg and std must have matching [time, feature] shapes')
        if avg.shape[1] != x.shape[-1]:
            raise ValueError(f'{error}: feature count does not match input')
        if not torch.isfinite(avg).all() or not torch.isfinite(std).all():
            raise ValueError(f'{error}: avg and std must be finite')
        if (std < 0).any():
            raise ValueError(f'{error}: std must be nonnegative')

        if self.name == '15m':
            # One statistic per intraday bar; the same statistics serve every day.
            intraday_bars = x.shape[-2]
            if avg.shape[0] != intraday_bars:
                raise ValueError(f'{error}: expected exactly {intraday_bars} intraday positions')
        elif self.name == '30mcont':
            # The special handler flattens days and intraday bars into one sequence.
            days, intraday_bars = x.shape[-3:-1]
            required_positions = days * intraday_bars
            if avg.shape[0] < required_positions:
                raise ValueError(f'{error}: need at least {required_positions} flattened positions')
        else:
            # Day data uses the trailing portion of the fitted historical window.
            window_length = x.shape[-2]
            if avg.shape[0] < window_length:
                raise ValueError(f'{error}: need at least {window_length} sequence positions')

    @classmethod
    def from_input(cls, name : str , prenorm_method: dict[str, Any] | None = None):
        prenorm_method = prenorm_method or {}
        return cls(
            name = name ,
            divlast = prenorm_method.get('divlast'  , False) and PreProHistNorm.is_divlast(name) ,
            histnorm = prenorm_method.get('histnorm' , False) and PreProHistNorm.has_histnorm(name) ,
            channelnorm = prenorm_method.get('channelnorm' , False)
        )

class SpecialDataPrenorm:
    """special prenorm for special data types, if defined will override other prenorm methods"""
    @classmethod
    def sp_30mcont(cls , x : torch.Tensor , histnorm : PreProHistNorm | None = None , features : Base.alias.NamesType = None, *, divlast: bool = True) -> torch.Tensor:
        assert features is not None , 'features is required for 30mcont'
        assert x.ndim == 4 , f'x must be 4-dimensional for 30mcont, but got {x.shape}'
        last_day_sum = x[:, -1].sum(dim=-2, keepdim=True)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        price_idx = torch.tensor([feat in ('close' , 'high' , 'low' , 'open' , 'vwap') for feat in Base.ensure_name_list(features , [])])

        # divlast
        if divlast:
            x_div = x[:,-1:].clone()
            x_div[...,~price_idx] = last_day_sum[...,~price_idx]
            x = x / (x_div + 1e-6)

        # histnorm
        if histnorm is not None:
            x = x - histnorm.avg[-x.shape[-2]:]
            x = x / (histnorm.std[-x.shape[-2]:] + 1e-6)
            if x.shape[-2] > 1:
                x[:,-1,price_idx] = 0
        return x

    @classmethod
    def sp_15mraw(cls , x : torch.Tensor , histnorm : PreProHistNorm | None = None , features : Base.alias.NamesType = None) -> torch.Tensor:
        assert features is not None , 'features is required for 15mraw'
        assert x.ndim == 4 , f'x must be 4-dimensional for 15mraw, but got {x.shape}'

        # price divlast
        price_idx = torch.tensor([feat in ('close' , 'high' , 'low' , 'open' , 'vwap') for feat in Base.ensure_name_list(features , [])])
        x[...,price_idx] = x[...,price_idx] / (x[:,-1:,-1:,price_idx] + 1e-6)

        # volume divsum
        x[...,~price_idx] = x[...,~price_idx] / (x[...,~price_idx].sum(dim = (1,2), keepdim = True) + 1e-6)
        return x

class PrenormOperator:
    """prenorm operator for data module datas"""
    def __init__(self, config: ModelConfig , histnorms : dict[str, PreProHistNorm] | None = None):
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
