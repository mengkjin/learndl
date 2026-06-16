"""Data containers for the boost pipeline.

Classes:
    BoostOutput      — flat prediction container with secid/date index
    BoostInput       — aligned 3-D tensor container with weight computation
    BoostWeightMethod — three-axis (ts/cs/bm) sample weight calculator
"""
from __future__ import annotations
import torch
import numpy as np

from dataclasses import dataclass
from typing import Literal

__all__ = ['BoostWeightMethod']
   
@dataclass(slots=True)
class BoostWeightMethod:
    """Three-axis multiplicative sample weight calculator.

    Computes ``w = cs_weight * ts_weight * bm_weight`` element-wise over a
    ``(n_sample, n_date)`` grid.

    Attributes:
        ts_type:           Time-series weighting scheme.
                           ``'lin'`` — linearly increasing from ``ts_lin_rate``
                           to 1 across dates.
                           ``'exp'`` — exponential decay with half-life
                           ``ts_half_life_rate * n_date``.
        cs_type:           Cross-sectional weighting scheme.
                           ``'ones'`` — doubles weight on positive-label samples.
                           ``'top'`` — exponential rank-based upweighting.
        bm_type:           Benchmark membership weighting.
                           ``'in'`` — doubles weight for securities in
                           ``bm_secid``.
        ts_lin_rate:       Start value of linear time-series weights (default 0.5).
        ts_half_life_rate: Half-life as a fraction of ``n_date`` (default 0.5).
        cs_top_tau:        Decay exponent for the ``'top'`` cross-sectional scheme.
        cs_ones_rate:      Multiplier applied to positive-label rows (default 2.0).
        bm_rate:           Multiplier applied to benchmark members (default 2.0).
        bm_secid:          Security IDs that constitute the benchmark universe.
    """
    ts_type : Literal['lin', 'exp'] | None = None
    cs_type : Literal['top', 'positive', 'ones'] | None = None
    bm_type : Literal['in'] | None = None
    ts_lin_rate : float = 0.5
    ts_half_life_rate : float = 0.5
    cs_top_tau : float = 0.75*np.log(0.5)/np.log(0.75)
    cs_ones_rate : float = 2.
    bm_rate : float = 2.
    bm_secid : np.ndarray | list | None = None

    def calculate_weight(self , y : np.ndarray | torch.Tensor , secid : np.ndarray) -> torch.Tensor:
        """Compute the combined ``(n_sample, n_date)`` weight matrix.

        The result is the element-wise product of :meth:`cs_weight`,
        :meth:`ts_weight`, and :meth:`bm_weight`.
        """
        if y.ndim == 3 and y.shape[-1] == 1:
            y = y[...,0]
        assert y.ndim == 2 , y.shape
        value = self.cs_weight(y) * self.ts_weight(y) * self.bm_weight(y , secid)
        if isinstance(value , torch.Tensor):
            return value
        else:
            return torch.from_numpy(value)

    def cs_weight(self , y : np.ndarray | torch.Tensor , **kwargs):
        """Cross-sectional weights of shape ``(n_sample, n_date)``.

        ``'ones'``: samples with label ``== 1`` get weight ``cs_ones_rate``
        (default * 2).
        ``'top'``: exponential rank-based decay so top-ranked securities receive
        higher weight.  ``None``: uniform ones.
        """
        w = y * 0 + 1.
        if self.cs_type is None: 
            return w
        elif self.cs_type == 'ones':
            w[y == 1.] = w[y == 1.] * 2
        elif self.cs_type == 'top':
            for j in range(w.shape[1]):
                v = y[:,j].argsort() + y[:,j] * 0
                w[:,j] = np.exp((1 - v / np.nanmax(v).astype(float))*np.log(0.5) / self.cs_top_tau)
        else:
            raise KeyError(self.cs_type)
        return w
    
    def ts_weight(self , y : np.ndarray | torch.Tensor , **kwargs):
        """Time-series weights of shape ``(n_sample, n_date)``.

        ``'lin'``: linearly increases from ``ts_lin_rate`` to 1 across dates.
        ``'exp'``: exponential decay so recent dates have higher weight.
        ``None``: uniform ones.
        """
        w = y * 0 + 1.
        if self.ts_type is None: 
            return w
        elif self.ts_type == 'lin':
            w *= np.linspace(self.ts_lin_rate,1,w.shape[1]).reshape(1,-1)
        elif self.ts_type == 'exp':
            w *= np.power(2 , -np.arange(w.shape[1])[::-1] / int(self.ts_half_life_rate * w.shape[1])).reshape(1,-1)
        else:
            raise KeyError(self.ts_type)
        return w
    
    def bm_weight(self , y : np.ndarray | torch.Tensor , secid : np.ndarray | list):
        """Benchmark-membership weights of shape ``(n_sample, n_date)``.

        ``'in'``: securities in ``bm_secid`` receive weight ``bm_rate + 1``
        (default ×2), others weight 1.  ``None``: uniform ones.
        """
        w = y * 0 + 1.
        if self.bm_type is None: 
            return w
        elif self.bm_type == 'in': 
            if self.bm_secid is not None:
                w *= np.isin(secid , self.bm_secid) * 1 + 1
        else:
            raise KeyError(self.bm_type)
        return w
    
    def reset(self , **kwargs):
        [setattr(self , k , v) for k,v in kwargs.items()]