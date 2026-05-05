"""Accuracy metric registry for NN evaluation.

Accuracy metrics are higher-is-better (sign-flipped compared to their
corresponding loss functions where applicable).

Registry keys: 'mse', 'pearson', 'ccc', 'spearman'
"""
import torch
import torch.nn as nn

from src.func.metric import mse , pearson , ccc , spearman

from .basic import align_shape, first_output

__all__ = ['Accuracy']

class BaseAccuracy(nn.Module):
    """Abstract base class for accuracy metrics.

    Parallel structure to ``BaseLoss`` but higher values indicate better
    performance.  Subclasses must set ``key`` and implement ``forward()``.
    """
    key : str = ''
    def __init__(self , **kwargs):
        super().__init__()

    def __call__(self , *args , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        return self.forward(*args , **kwargs)

    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor | dict[str,torch.Tensor]:
        raise NotImplementedError

class MSEAccuracy(BaseAccuracy):
    """Negative MSE accuracy: ``-mse``.  Higher is better."""
    key = 'mse'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return -mse(*align_shape(label , pred , w) , dim)

class Pearson(BaseAccuracy):
    """Pearson correlation accuracy.  Higher is better (range: [-1, 1])."""
    key = 'pearson'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return pearson(*align_shape(label , pred , w) , dim)

class CCC(BaseAccuracy):
    """Concordance Correlation Coefficient (CCC) accuracy.  Higher is better."""
    key = 'ccc'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return ccc(*align_shape(label , pred , w) , dim)

class Spearman(BaseAccuracy):
    """Spearman rank correlation accuracy.  Higher is better (range: [-1, 1])."""
    key = 'spearman'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        return spearman(*align_shape(label , pred , w) , dim)

class LongAvg(BaseAccuracy):
    """The average of the top 5% of predictions ."""
    key = 'long_avg'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        label , pred , w = align_shape(label , pred , w)
        pred0 = first_output(pred)
        label0 = first_output(label).nan_to_num(0)
        top_5_pct_mask = pred0.topk(int(len(pred0) * 0.05), dim = 0 , sorted = False).indices
        return label0[top_5_pct_mask].mean()

class LongShortDiff(BaseAccuracy):
    """The average of the top 5% of predictions ."""
    key = 'long_short'
    def forward(self , label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None , dim = None , **kwargs) -> torch.Tensor:
        label , pred , w = align_shape(label , pred , w)
        pred0 = first_output(pred)
        label0 = first_output(label).nan_to_num(0)
        top_5_pct_mask = pred0.topk(int(len(pred0) * 0.05), dim = 0 , largest = True , sorted = False).indices
        bot_5_pct_mask = pred0.topk(int(len(pred0) * 0.05), dim = 0 , largest = False , sorted = False).indices
        return label0[top_5_pct_mask].mean() - label0[bot_5_pct_mask].mean()

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