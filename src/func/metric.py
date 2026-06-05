"""Weighted and rank-based correlation / loss helpers on torch tensors (optional ``dim`` reduction)."""
from __future__ import annotations
import torch
import numpy as np

from scipy import stats

from .basic import DIV_TOL
from .tensor import rank_pct , rank

def rank_weight(x : torch.Tensor , dim = None):
    """Turn ranks into normalized weights peaked at the largest rank.

    Args:
        x: Input tensor.
        dim: Axis for ranking and normalization; ``None`` reduces globally.

    Returns:
        Non-negative weights summing to 1 along ``dim`` (or globally).
    """
    r = rank(x , dim = dim)
    n = (~r.isnan()).sum(dim=dim,keepdim=True)
    p = (n - 1 - r) * 2 / (n - 1)
    w = torch.pow(0.5,p)
    return w / w.sum()

def demean(x : torch.Tensor, w : torch.Tensor | float, dim = None):
    """Subtract a weighted mean from ``x``.

    Args:
        x: Values.
        w: Weights (broadcastable) or scalar.
        dim: Reduction axis for the mean; ``None`` for full tensor.

    Returns:
        ``x`` minus weighted mean with ``keepdim=True`` semantics inside the mean.
    """
    return x - (w * x).mean(dim=dim,keepdim=True)

def pearson(x : torch.Tensor, y : torch.Tensor , w = None, dim = None , **kwargs):
    """Weighted Pearson correlation along ``dim`` (or global if ``dim`` is None)."""
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    x1 , y1 = demean(x , w , dim) , demean(y , w , dim)
    return (w * x1 * y1).mean(dim = dim) / ((w * x1.square()).mean(dim=dim).sqrt() + DIV_TOL) / ((w * y1.square()).mean(dim=dim).sqrt() + DIV_TOL)

def ccc(x : torch.Tensor , y : torch.Tensor , w = None, dim = None , **kwargs):
    """Lin's concordance correlation coefficient (weighted).

    Args:
        x: First variable.
        y: Second variable.
        w: Optional weights.
        dim: Reduction axis.
        **kwargs: Ignored.

    Returns:
        CCC values along ``dim`` or scalar.
    """
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    x1 , y1 = demean(x , w , dim) , demean(y , w , dim)
    cov_xy = (w * x1 * y1).mean(dim=dim)
    mse_xy = (w * (x1 - y1).square()).mean(dim=dim)
    return (2 * cov_xy) / (mse_xy + 2 * cov_xy + DIV_TOL)

def mse(x : torch.Tensor , y : torch.Tensor , w = None, dim = None , reduction='mean' , **kwargs):
    """Weighted mean squared error.

    Args:
        x: Predictions.
        y: Targets.
        w: Optional weights.
        dim: Reduction axis.
        reduction: ``'mean'`` or ``'sum'``.
        **kwargs: Ignored.

    Returns:
        MSE reduced per ``reduction`` along ``dim`` (or global).
    """
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    f = torch.mean if reduction == 'mean' else torch.sum
    return f(w * (x - y).square() , dim=dim)

def spearman(x : torch.Tensor , y : torch.Tensor , w = None , dim = None , **kwargs):
    """Spearman rank correlation via ranking then Pearson.

    Args:
        x: First variable.
        y: Second variable.
        w: Optional weights for Pearson step.
        dim: Axis for ranks and correlation.
        **kwargs: Passed to ``pearson``.

    Returns:
        Spearman correlation.
    """
    x , y = rank(x , dim = dim) , rank(y , dim = dim)
    return pearson(x , y , w , dim , **kwargs)

def wpearson(x : torch.Tensor , y : torch.Tensor , dim = None , **kwargs):
    """Pearson with weights from ``rank_weight(y, dim)``.

    Args:
        x: First variable.
        y: Second variable (defines weights via ranks).
        dim: Axis for weights and correlation.
        **kwargs: Passed to ``pearson``.

    Returns:
        Weighted Pearson correlation.
    """
    w = rank_weight(y , dim = dim)
    return pearson(x,y,w,dim)

def wccc(x : torch.Tensor , y : torch.Tensor , dim = None , **kwargs):
    """CCC with ``rank_weight(y)`` weights.

    Args:
        x: First variable.
        y: Second variable.
        dim: Axis.
        **kwargs: Passed to ``ccc``.

    Returns:
        Weighted CCC.
    """
    w = rank_weight(y , dim = dim)
    return ccc(x,y,w,dim)

def wmse(x : torch.Tensor , y : torch.Tensor , dim = None , reduction='mean' , **kwargs):
    """MSE with ``rank_weight(y)`` weights.

    Args:
        x: Predictions.
        y: Targets.
        dim: Axis.
        reduction: ``'mean'`` or ``'sum'``.
        **kwargs: Passed to ``mse``.

    Returns:
        Weighted MSE.
    """
    w = rank_weight(y , dim = dim)
    return mse(x,y,w,dim,reduction)

def wspearman(x : torch.Tensor , y : torch.Tensor , dim = None , **kwargs):
    """Spearman with ``rank_weight(y)`` in the Pearson step.

    Args:
        x: First variable.
        y: Second variable.
        dim: Axis.
        **kwargs: Passed to ``spearman``.

    Returns:
        Weighted Spearman correlation.
    """
    w = rank_weight(y , dim = dim)
    return spearman(x,y,w,dim)

def np_drop_na(x : np.ndarray , y : np.ndarray):
    """Drop positions where either ``x`` or ``y`` is NaN (pairwise).

    Args:
        x: 1-D array.
        y: 1-D array, same length as ``x``.

    Returns:
        Tuple ``(x_clean, y_clean)`` with NaN pairs removed.
    """
    pairwise_nan = np.isnan(x) + np.isnan(y)
    x , y = x[~pairwise_nan] , y[~pairwise_nan]
    return x , y

def np_ic(x : np.ndarray , y : np.ndarray):
    """Scalar Pearson correlation after dropping pairwise NaNs.

    Args:
        x: 1-D array.
        y: 1-D array.

    Returns:
        Correlation coefficient or ``nan`` on failure.
    """
    x , y = np_drop_na(x,y)
    try:
        return stats.pearsonr(x,y)[0]
    except Exception:
        return np.nan
    
def np_rankic(x : np.ndarray , y : np.ndarray):
    """Scalar Spearman correlation after dropping pairwise NaNs.

    Args:
        x: 1-D array.
        y: 1-D array.

    Returns:
        Correlation coefficient or ``nan`` on failure.
    """
    x , y = np_drop_na(x,y)
    try:
        return stats.spearmanr(x,y)[0]
    except Exception:
        return np.nan
    
def np_ic_2d(x : np.ndarray , y : np.ndarray , dim=0):
    """Pearson correlation along an axis (NaN-aware).

    Args:
        x: 2-D array.
        y: 2-D array, broadcast-aligned with ``x``.
        dim: ``0`` correlates across rows per column; ``1`` across columns per row.

    Returns:
        1-D array of correlations along the orthogonal axis.
    """
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - np.nanmean(x, dim, keepdims=True)  
    y_ymean = y - np.nanmean(y, dim, keepdims=True) 
    cov  = np.nansum(x_xmean * y_ymean, dim) 
    ssd  = (np.nansum(np.square(x_xmean), dim) ** 0.5) * (np.nansum(np.square(y_ymean), dim) ** 0.5 )
    ssd[ssd == 0] = 1e-4
    corr = cov / ssd
    return corr

def np_rankic_2d(x : np.ndarray , y : np.ndarray , dim = 0):
    """Spearman IC per slice: loop over columns (``dim=0``) or rows (``dim=1``).

    Args:
        x: 2-D array.
        y: 2-D array.
        dim: ``0`` for column-wise 1-D rank ICs; ``1`` for row-wise.

    Returns:
        1-D ``np.array`` of per-slice Spearman correlations.
    """
    if dim == 0:
        return np.array([np_rankic(x[:,i],y[:,i]) for i in range(x.shape[1])])
    else:
        return np.array([np_rankic(x[i,:],y[i,:]) for i in range(x.shape[0])])

def ic(x : torch.Tensor , y : torch.Tensor):
    """Flattened Pearson IC between two tensors.

    Args:
        x: Tensor.
        y: Tensor, same shape as ``x``.

    Returns:
        Scalar Pearson correlation along the flattened dimension.
    """
    return ic_2d(x.flatten() , y.flatten() , 0)

def rankic(x : torch.Tensor , y : torch.Tensor):
    """Flattened rank IC between two tensors.

    Args:
        x: Tensor.
        y: Tensor, same shape as ``x``.

    Returns:
        Rank IC computed on flattened inputs (see ``rankic_2d`` with ``dim=0``).
    """
    return rankic_2d(x.flatten() , y.flatten() , 0)

def ic_2d(x : torch.Tensor , y : torch.Tensor , dim=0):
    """Pearson correlation along ``dim`` (NaN-aware, broadcast alignment).

    Args:
        x: Tensor.
        y: Tensor, aligned with ``x``.
        dim: Axis along which means are taken; correlation reduces this axis.

    Returns:
        Correlation tensor with ``dim`` reduced.
    """
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  
    y_ymean = y - torch.nanmean(y, dim, keepdim=True) 
    cov  = torch.nansum(x_xmean * y_ymean, dim) 
    ssd  = x_xmean.square().nansum(dim).sqrt() * y_ymean.square().nansum(dim).sqrt()
    ssd[ssd == 0] = 1e-4
    corr = cov / ssd
    return corr

def rankic_2d(x : torch.Tensor, y : torch.Tensor , dim = 1 , universe = None , min_coverage = 0.5):
    """Rank correlation with coverage masking.

    Args:
        x: Tensor (same ndim as ``y``).
        y: Tensor; NaNs define invalid positions unless covered by ``universe``.
        dim: Axis for rank_pct and correlation.
        universe: Optional boolean mask; intersected with finite ``y`` for validity.
        min_coverage: Mask IC to NaN where valid ``x`` count falls below this fraction of valid ``y`` count.

    Returns:
        Rank IC tensor with insufficient-coverage entries set to NaN.
    """
    valid = ~y.isnan()
    if universe is not None: 
        valid *= universe.nan_to_num(False)
    x = torch.where(valid , x , torch.nan)

    coverage = (~x.isnan()).sum(dim=dim)
    x = rank_pct(x , dim = dim)
    y = rank_pct(y , dim = dim)
    ic = ic_2d(x , y , dim=dim)
    return ic if ic is None else torch.where(coverage < min_coverage * valid.sum(dim=dim) , torch.nan , ic)
