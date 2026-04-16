"""Reversible Instance Normalization (RevIN) for time-series models.

Reference: Kim et al. (2022) "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift."
"""
import torch
from torch import nn , Tensor

class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Normalizes the input in the ``'norm'`` mode and inverts the normalization
    in the ``'denorm'`` mode, enabling a model to operate on standardized
    sequences while producing outputs in the original scale.

    Args:
        num_features:  Number of channels / features (last dimension of input).
        eps:           Small constant added to variance for numerical stability.
        affine:        If ``True``, add learnable affine parameters (scale and
                       shift) after normalization.
        subtract_last: If ``True``, subtract the last time step value instead
                       of the temporal mean.  Useful for trend-removal.

    Usage::

        revin = RevIN(num_features)
        x_norm = revin(x, 'norm')
        y_norm = model(x_norm)
        y = revin(y_norm, 'denorm')
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x : Tensor, mode:str) -> Tensor:
        """Apply normalization or denormalization.

        Args:
            x:    Input tensor of shape ``[bs, seq_len, num_features]``.
            mode: ``'norm'``   — compute statistics from ``x`` and normalize;
                  ``'denorm'`` — invert a previously stored normalization.

        Returns:
            Transformed tensor with the same shape as ``x``.
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """Compute and cache mean (or last step) and standard deviation of x.

        Reduces over all dims except the first (batch) and last (feature).
        Statistics are detached from the computation graph so they do not
        accumulate gradients.
        """
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:]
        else:
            self.mean = torch.mean(x, dim=dim2reduce).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, unbiased=False) + self.eps).detach()
        
    def _normalize(self, x):
        shp = self._stat_shape(x)
        if self.subtract_last:
            x = x - self.last.reshape(*shp)
        else:
            x = x - self.mean.reshape(*shp)
        x = x / self.stdev.reshape(*shp)
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        shp = self._stat_shape(x)
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev.reshape(*shp)
        if self.subtract_last:
            x = x + self.last.reshape(*shp)
        else:
            x = x + self.mean.reshape(*shp)
        return x
    
    def _stat_shape(self , x):
        """Return a reshape spec for broadcasting statistics onto x.

        Returns a list ``[bs, 1, 1, ..., -1]`` where the number of ``1``s
        equals ``x.ndim - 2``, ensuring that per-feature statistics broadcast
        correctly along all inner (sequence) dimensions.
        """
        return [x.shape[0] , *[1 for _ in range(x.ndim - 2)] , -1]