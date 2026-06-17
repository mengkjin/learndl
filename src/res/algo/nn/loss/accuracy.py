"""Accuracy metric registry for NN evaluation.

Accuracy metrics are higher-is-better (sign-flipped compared to their
corresponding loss functions where applicable).

Registry keys: 'mse', 'pearson', 'ccc', 'spearman'
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Literal , Any , TypeAlias

from src.func.metric import mse , pearson , ccc , spearman

from .basic import first_output , mask_topx

__all__ = ['Accuracy']

class BaseAccuracy(nn.Module):
    """Abstract base class for accuracy metrics.

    Parallel structure to ``BaseLoss`` but higher values indicate better
    performance.  Subclasses must set ``key`` and implement ``forward()``.
    """
    key : str = ''
    def __init__(self , **kwargs):
        super().__init__()

    def __call__(self , *args , **kwargs) -> torch.Tensor | dict[str, torch.Tensor]:
        return self.forward(*args , **kwargs)

    def forward(
        self , pred : torch.Tensor , label : torch.Tensor , weight : torch.Tensor | None = None , 
        dim : int | None = None , **kwargs
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        raise NotImplementedError

class MSEAccuracy(BaseAccuracy):
    """Negative MSE accuracy: ``-mse``.  Higher is better."""
    key = 'mse'
    def forward(
        self , pred : torch.Tensor , label : torch.Tensor , weight : torch.Tensor | None = None , 
        dim : int | None = None , **kwargs
    ) -> torch.Tensor:
        return -mse(pred , label , weight , dim)

class Pearson(BaseAccuracy):
    """Pearson correlation accuracy.  Higher is better (range: [-1, 1])."""
    key = 'pearson'
    def forward(
        self , pred : torch.Tensor , label : torch.Tensor , weight : torch.Tensor | None = None , 
        dim : int | None = None , **kwargs
    ) -> torch.Tensor:
        return pearson(pred , label , weight , dim)

class CCC(BaseAccuracy):
    """Concordance Correlation Coefficient (CCC) accuracy.  Higher is better."""
    key = 'ccc'
    def forward(
        self , pred : torch.Tensor , label : torch.Tensor , weight : torch.Tensor | None = None , 
        dim : int | None = None , **kwargs
    ) -> torch.Tensor:
        return ccc(pred , label , weight , dim)

class Spearman(BaseAccuracy):
    """Spearman rank correlation accuracy.  Higher is better (range: [-1, 1])."""
    key = 'spearman'
    def forward(
        self , pred : torch.Tensor , label : torch.Tensor , weight : torch.Tensor | None = None , 
        dim : int | None = None , **kwargs
    ) -> torch.Tensor:
        return spearman(pred , label , weight , dim)

class LongAvg(BaseAccuracy):
    """The average of the top 5% of predictions ."""
    key = 'long_avg'
    def forward(
        self , pred : torch.Tensor , label : torch.Tensor , weight : torch.Tensor | None = None , 
        dim : int | None = None , **kwargs
    ) -> torch.Tensor:
        top_pred = mask_topx(pred , 0.05 , dim = 0 , fill_nan = True)
        label0 = first_output(label).unsqueeze(-1).nan_to_num(0)
        return (label0 * top_pred).nanmean(dim = dim)

class LongShortDiff(BaseAccuracy):
    """The average of the top 5% of predictions ."""
    key = 'long_short'
    def forward(
        self , pred : torch.Tensor , label : torch.Tensor , weight : torch.Tensor | None = None , 
        dim : int | None = None , **kwargs
    ) -> torch.Tensor:
        top_pred = mask_topx(pred , 0.05 , dim = 0 , fill_nan = True)
        bot_pred = mask_topx(pred , 0.05 , dim = 0 , fill_nan = True , ascending = False)
        label0 = first_output(label).unsqueeze(-1).nan_to_num(0)
        return (label0 * top_pred).nanmean(dim = dim) - (label0 * bot_pred).nanmean(dim = dim)

class ProgressiveGlobal2Top(BaseAccuracy):
    """Progressive Global to Top-K loss."""
    key = 'global2top'
    GlobalOptionType : TypeAlias = Literal['mse' , 'pearson', 'ccc' , 'spearman']
    TopOptionType : TypeAlias = Literal['long_avg' , 'long_short']
    def __init__(self, 
        global_option : GlobalOptionType = 'spearman', 
        top_option : TopOptionType = 'long_avg', 
        global_kwargs : dict[str, Any] | None = None , 
        top_kwargs : dict[str, Any] | None = None , 
        base_top_lambda : float = 1. ,
    ):
        super().__init__()
        assert base_top_lambda > 0 , f'base_top_lambda must be positive'
        self.global_option = global_option
        self.top_option = top_option
        match global_option:
            case 'spearman':
                self.global_accu = Spearman(**(global_kwargs or {}))
            case 'pearson':
                self.global_accu = Pearson(**(global_kwargs or {}))
            case 'ccc':
                self.global_accu = CCC(**(global_kwargs or {}))
            case 'mse':
                self.global_accu = MSEAccuracy(**(global_kwargs or {}))
            case _:
                raise ValueError(f'Invalid global option: {global_option}')
        
        match top_option:
            case 'long_avg':
                self.top_accu = LongAvg(**(top_kwargs or {}))
            case 'long_short':
                self.top_accu = LongShortDiff(**(top_kwargs or {}))
            case _:
                raise ValueError(f'Invalid top option: {top_option}')
        self.base_top_lambda = base_top_lambda

    def forward(
        self, pred : torch.Tensor , label : torch.Tensor , weight : torch.Tensor | None = None , 
        dim : int | None = None , **kwargs
    ) -> dict[str,torch.Tensor]:
        global_accu = self.global_accu(pred , label , weight , dim , **kwargs)
        top_accu   = self.top_accu(pred , label , weight , dim , **kwargs)
        assert isinstance(global_accu, torch.Tensor) and isinstance(top_accu, torch.Tensor) , f'global_accu and top_accu should be tensors'
        return {
            f'global.{self.global_option}' : global_accu ,
            f'top.{self.top_option}' : self.base_top_lambda * top_accu
        }

class Accuracy:
    """Factory for ``BaseAccuracy`` instances.

    ``options`` is populated at class definition time from all direct
    ``BaseAccuracy`` subclasses with a non-empty ``key``.

    Usage::

        acc_fn = Accuracy.get('spearman')
    """
    @classmethod
    def options(cls) -> dict[str, type[BaseAccuracy]]:
        if not hasattr(cls, '_options'):
            options = {}
            for subclass in BaseAccuracy.__subclasses__():
                if not subclass.key:
                    continue
                if subclass.key in options:
                    raise ValueError(f'{subclass.__name__}.key {subclass.key} is already registered, check for duplication')
                options[subclass.key] = subclass
            cls._options = options
        return cls._options

    @classmethod
    def get(cls , name : str , **kwargs) -> BaseAccuracy:
        """Return an instantiated accuracy metric by registry key.

        Args:
            name:    Registry key (e.g. ``'pearson'``, ``'spearman'``).
            **kwargs: Forwarded to the metric constructor.

        Returns:
            An instantiated ``BaseAccuracy`` subclass.
        """
        return cls.options()[name](**kwargs)