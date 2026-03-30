"""Torch-first factor math: rolling windows, ranks, neutralization, and cross-sectional ops.

Conventions: tensors are often shaped ``[secid, time, feature]`` (or 2-D slices). The ``dim`` argument
typically selects the cross-section axis (default ``0``) for ``process_factor`` / ``standardize`` / ``rank``;
for time rolling, ``TsRoller`` defaults to ``dim=1`` (time). ``dim`` is the axis reduced or along which
statistics are taken, depending on the function.
"""

import torch
import torch.nn.functional as F
from torch import Tensor , nan
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Literal , Callable

from .basic import alert_message , DIV_TOL , allna
    
def process_factor(value : Tensor | None , * , stream = 'inf_winsor_norm' , dim = 0 , trim_ratio = 7. , **kwargs):
    """Apply a chained underscore-separated pipeline to factor values (trim, winsor, norm, etc.).

    Supported tokens include ``mean``, ``inf``, ``trim``, ``winsor``, ``norm``, ``nan``. ``trim``/``winsor``
    use cross-sectional medians and IQR-based bandwidth along ``dim``; ``trim_ratio`` scales half-IQR width.

    Args:
        value: Factor tensor or None; None or all-NaN returns None.
        stream: Underscore-separated step names.
        dim: Axis for cross-sectional stats (default 0).
        trim_ratio: Outlier half-width multiplier for trim/winsor.
        **kwargs: Reserved for forward compatibility.

    Returns:
        Processed tensor or None.
    """
    if value is None or allna(value , inf_as_na = True): 
        return None
    
    # assert 'inf' in stream or 'trim' in stream or 'winsor' in stream , stream
    if 'trim' in stream or 'winsor' in stream:
        med       = value.nanquantile(0.5, dim=dim,keepdim = True , interpolation='lower')
        bandwidth = (value.nanquantile(0.75 , dim , keepdim=True) - value.nanquantile(0.25 , dim , keepdim=True)) / 2
        lbound , ubound = med - trim_ratio * bandwidth , med + trim_ratio * bandwidth
    for _str in stream.split('_'):
        if _str == 'mean':
            value -= torch.nanmean(value , dim, keepdim=True)
        elif _str == 'inf':
            value.nan_to_num_(torch.nan,torch.nan,torch.nan)
        elif _str == 'trim':
            value[(value > ubound) + (value < lbound)] = torch.nan
        elif _str == 'winsor':
            value = torch.where(value > ubound , ubound , value)
            value = torch.where(value < lbound , lbound , value)
        elif _str == 'norm': 
            value -= torch.nanmean(value , dim, keepdim=True)
            value /= value.square().nansum(dim , keepdim = True).sqrt() + 1e-6 
        elif _str == 'nan': 
            value = value.nan_to_num_()
    return value

class TsRoller:
    """Time-series rolling via ``unfold`` / ``fold`` along an axis (default ``dim=1`` time).

    Decorators wrap unary or binary kernels so they receive window tensors with the window axis last
    (``dim=-1`` inside the kernel after unfold).
    """

    @staticmethod
    def unfold(x : Tensor , d : int , * , dim :int | Literal[1] = 1, nan = nan , pinf = torch.inf , ninf = -torch.inf, **kwargs):
        """Build sliding windows of length ``d`` along ``dim``.

        Args:
            x: Input tensor.
            d: Window size (>= 1).
            dim: Axis to unfold (typically time).
            nan, pinf, ninf: Fill values for ``nan_to_num`` before masking.
            **kwargs: Reserved.

        Returns:
            Tensor where each position along ``dim`` holds a length-``d`` slice; invalid windows are NaN.
        """
        unfold = x.unfold(dim,d,1)
        valid = unfold.sum(dim = -1 , keepdim = True).isfinite()
        return unfold.nan_to_num(nan,pinf,ninf).where(valid , torch.nan)
    
    @staticmethod
    def fold(z : Tensor , d : int , * , dim : int = 1 , **kwargs):
        """Pad and reshape rolling output back to the shape before ``unfold``.

        Args:
            z: Kernel output aligned with ``unfold`` layout.
            d: Window length (must match ``unfold``).
            dim: Same axis as used in ``unfold``.
            **kwargs: Passed padding uses ``nan`` as pad value where applicable.

        Returns:
            Tensor with original-like layout along ``dim``.
        """
        pad = tuple([0] * (z.ndim - dim - 1) * 2 + [d-1,0])
        return F.pad(z , pad , value = nan).nan_to_num(nan)

    @staticmethod
    def unfold_chunk_slice_x(chunk_size = 8e8):
        """Decorator: run a unary rolling kernel on time chunks to limit memory (assumes ``dim=1`` slicing).

        Args:
            chunk_size: Approximate element budget ``prod(shape)*d`` per chunk.

        Returns:
            A decorator that wraps ``func(x, d, ...)`` and concatenates chunk outputs along dim 1.
        """
        def decorator(func : Callable[..., Tensor]):
            """Wrap a unary rolling ``func`` for chunked time-axis execution."""
            def wrapper(x : Tensor , d : int , *args , **kwargs):
                """Run ``func`` on time chunks of ``x`` and concatenate along dim 1."""
                chunk_num = (np.prod(x.shape) * d / chunk_size).__ceil__()
                chunk_len = ((x.shape[1] + (chunk_num - 1) * d) / chunk_num).__ceil__()
                sub_rets : list[Tensor] = []
                for i in range(chunk_num):
                    start = max(i * chunk_len - d , 0)
                    end   = (i + 1) * chunk_len
                    sub_rets.append(func(x[:,start:end] , d , *args , **kwargs)[:,d if i > 0 else 0:])
                ret = torch.concat(sub_rets , dim = 1)
                return ret
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

    @staticmethod
    def unfold_chunk_slice_xy(chunk_size = 4e8):
        """Like ``unfold_chunk_slice_x`` but for ``func(x, y, d, ...)`` (paired time slices).

        Args:
            chunk_size: Memory budget heuristic (smaller default than unary).

        Returns:
            Decorator concatenating results on dim 1.
        """
        def decorator(func : Callable[..., Tensor]):
            """Wrap a binary rolling ``func`` for chunked time-axis execution."""
            def wrapper(x : Tensor , y : Tensor , d : int , *args , **kwargs):
                """Run ``func`` on aligned time chunks of ``x`` and ``y``; concat on dim 1."""
                chunk_num = (np.prod(x.shape) * d / chunk_size).__ceil__()
                chunk_len = ((x.shape[1] + (chunk_num - 1) * d) / chunk_num).__ceil__()
                sub_rets : list[Tensor] = []
                for i in range(chunk_num):
                    start = max(i * chunk_len - d , 0)
                    end   = (i + 1) * chunk_len
                    sub_rets.append(func(x[:,start:end] , y[:,start:end] , d , *args , **kwargs)[:,d if i > 0 else 0:])
                ret = torch.concat(sub_rets , dim = 1)
                return ret
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

    @classmethod
    def decor(cls , n_arg : Literal[1,2] = 1, **decor_kwargs):
        """Return ``decorator_x`` or ``decorator_xy`` with shared nan/inf kwargs.

        Args:
            n_arg: ``1`` for single-input rolling, ``2`` for two inputs.
            **decor_kwargs: Forwarded to ``unfold``/``fold`` (e.g. ``nan``).

        Returns:
            A decorator factory.

        Raises:
            ValueError: If ``n_arg`` is not 1 or 2.
        """
        if n_arg == 1:
            return cls.decorator_x(**decor_kwargs)
        elif n_arg == 2:
            return cls.decorator_xy(**decor_kwargs)
        else:
            raise ValueError(f'Invalid number of arguments: {n_arg}')

    @classmethod
    def decorator_x(cls , nan = nan , pinf = torch.inf , ninf = -torch.inf, **decor_kwargs):
        """Rolling wrapper: ``unfold`` → ``func(..., dim=-1)`` → ``fold``.

        Args:
            nan, pinf, ninf: Non-finite handling for unfold.
            **decor_kwargs: Extra args for ``unfold``/``fold``.

        Returns:
            A ``decorator`` that produces ``wrapper(x, d, *, dim=1, ...)``.
        """
        def decorator(func):
            """Return a rolling wrapper around unary ``func`` using unfold/fold."""
            def wrapper(x : Tensor , d : int , *args , dim : int = 1 , **kwargs):
                """Apply ``func`` inside length-``d`` windows along ``dim``."""
                x = cls.unfold(x , d , dim = dim , nan = nan , pinf = pinf , ninf = ninf, **decor_kwargs)
                z = func(x , 1 , *args , dim = -1 , **kwargs)
                z = cls.fold(z , d , dim = dim , nan = nan , pinf = pinf , ninf = ninf, **decor_kwargs)
                return z
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

    @classmethod
    def decorator_xy(cls , nan = nan , pinf = torch.inf , ninf = -torch.inf, **decor_kwargs):
        """Binary rolling wrapper: unfold ``x`` and ``y``, apply ``func``, then ``fold``.

        Args:
            nan, pinf, ninf: Non-finite handling.
            **decor_kwargs: Forwarded to ``unfold``/``fold``.

        Returns:
            Decorator factory for two-input rolling kernels.
        """
        def decorator(func):
            """Return a rolling wrapper around binary ``func`` using unfold/fold on ``x`` and ``y``."""
            def wrapper(x : Tensor , y : Tensor , d : int , *args , dim : int = 1 , **kwargs):
                """Apply ``func`` to paired windows of ``x`` and ``y`` along ``dim``."""
                x = cls.unfold(x , d , dim = dim , nan = nan , pinf = pinf , ninf = ninf, **decor_kwargs)
                y = cls.unfold(y , d , dim = dim , nan = nan , pinf = pinf , ninf = ninf, **decor_kwargs)
                z = func(x , y , 1 , *args , dim = -1 , **kwargs)
                z = cls.fold(z , d , dim = dim , nan = nan , pinf = pinf , ninf = ninf, **decor_kwargs)
                return z
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

def kthvalue_by_topk(x: Tensor, k: int, * , dim=-1, keepdim=True , largest=False):
    """k-th order statistic via ``torch.topk`` (sorted branch).

    Args:
        x: Input tensor.
        k: Which order statistic (1-based from smallest if ``largest=False``).
        dim: Axis to rank along.
        keepdim: If True, keep ``dim`` as size 1.
        largest: If False, k-th smallest; if True, k-th largest.

    Returns:
        Tensor of the k-th value along ``dim``.
    """
    # Get the k smallest elements
    vals, _ = torch.topk(x, k, dim=dim, largest=largest, sorted=True)
    # The k-th smallest is the last one in this sorted list
    res = vals.select(dim, -1)
    return res.unsqueeze(dim) if keepdim else res

def nanmean(x : torch.Tensor , * , dim=None, keepdim=False):
    """``torch.nanmean`` alias.

    Args:
        x: Input tensor.
        dim: Reduction axis or None for full tensor.
        keepdim: Whether to keep reduced dimensions.

    Returns:
        Nan-mean of ``x``.
    """
    return x.nanmean(dim = dim , keepdim = keepdim)
    
def nanstd(x : torch.Tensor , * , dim=None, keepdim=False , correction=1):
    """Sample standard deviation ignoring NaNs (Bessel ``correction``).

    Args:
        x: Input tensor.
        dim: Reduction axis.
        keepdim: Keep reduced dims on the std tensor.
        correction: Degrees of freedom subtracted (default 1).

    Returns:
        Nan-safe standard deviation.
    """
    x = x - torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = (torch.nansum(x.square(), dim , keepdim=keepdim) / (torch.sum(~x.isnan() ,dim = dim , keepdim=keepdim) - correction)).sqrt()
    return x_stddev

def nanmedian(x : torch.Tensor , dim=None, keepdim=False):
    """Nan-median via ``nanquantile(0.5)``.

    Args:
        x: Input tensor (cast to float if integer).
        dim: Reduction axis.
        keepdim: Keep reduced dimensions.

    Returns:
        Median tensor.
    """
    if not x.is_floating_point():
        x = x.to(torch.float)
    return x.nanquantile(0.5, dim=dim,keepdim = keepdim , interpolation='lower')

def standardize(x : torch.Tensor, * , dim : int | None = 0):
    """Z-score along ``dim`` using nan-mean and nan-std (``correction=0``) plus ``DIV_TOL``.

    Args:
        x: Input tensor.
        dim: Axis for statistics; if all NaN, returns ``x`` unchanged.

    Returns:
        Standardized tensor.
    """
    if x.isnan().all().item(): 
        return x
    x = (x - nanmean(x,dim=dim,keepdim=True)) / (nanstd(x,dim=dim,correction=0,keepdim=True) + DIV_TOL)
    return x

def rank(x : Tensor , * , dim : int | None = 0) -> Tensor:
    """Average-rank style order (1-based), NaNs preserved.

    Args:
        x: Tensor with at most 3 dimensions.
        dim: Axis to rank along, or ``None`` to rank the flattened tensor.

    Returns:
        Float ranks; NaN positions stay NaN.

    Raises:
        AssertionError: If ``ndim > 3``.
    """
    assert (len(x.shape) <= 3) , x.shape
    if dim is None:
        old_shape = x.shape
        x_rank = rank(x.flatten() , dim = 0).reshape(old_shape)
    else:
        x_rank = x.argsort(dim=dim).argsort(dim=dim).to(torch.float32) + 1 # .where(~x.isnan() , nan)
        x_rank[x.isnan()] = nan
    return x_rank

def rank_pct(x : Tensor , * , dim : int | None = 0) -> Tensor:
    """Rank divided by count of finite values along ``dim`` (percentile rank).

    Args:
        x: Tensor, at most 3-D.
        dim: Axis for ranking, or ``None`` for global.

    Returns:
        Rank percentages in ``(0, 1]`` scale (per finite count); NaNs preserved.

    Raises:
        AssertionError: If ``ndim > 3``.
    """
    assert (len(x.shape) <= 3) , x.shape
    if dim is None:
        old_shape = x.shape
        x_rank = rank_pct(x.flatten() , dim = 0).reshape(old_shape)
    else:
        x_rank = x.argsort(dim=dim).argsort(dim=dim).to(torch.float32) + 1 # .where(~x.isnan() , nan)
        x_rank[x.isnan()] = nan
        x_rank = x_rank / ((~x_rank.isnan()).sum(dim=dim, keepdim=True))
    return x_rank

def rankic_2d(x : Tensor , y : Tensor , * , dim : int | None = 0 , universe : Tensor | None = None , min_coverage = 0.5):
    """Rank correlation on 2-D tensors with coverage gating (uses ``corrwith`` on rank_pct).

    Args:
        x: 2-D tensor.
        y: 2-D tensor, same shape as ``x``.
        dim: Axis for rank and correlation.
        universe: Optional boolean mask for valid observations.
        min_coverage: Mask correlation to NaN where coverage is below this fraction.

    Returns:
        1-D or reduced correlation tensor along the orthogonal axis.

    Raises:
        AssertionError: If not both 2-D.
    """
    assert x.ndim == y.ndim == 2 , (x.shape , y.shape)
    valid = ~y.isnan()
    if universe is not None: 
        valid *= universe.nan_to_num(0).to(torch.bool)
    x = torch.where(valid , x , nan)

    coverage = (~x.isnan()).sum(dim=dim) / valid.sum(dim=dim)
    x = rank_pct(x , dim = dim)
    y = rank_pct(y , dim = dim)
    ic = corrwith(x , y , dim=dim)
    return torch.where(coverage < min_coverage, nan , ic)

def dummy(x : Tensor , * , ex_last = True):
    """One-hot encoding of integer-like group ids (memory heavy for many levels).

    Args:
        x: Group indices (tensor or convertible); NaNs mapped to a spare category.
        ex_last: If True, drop the last one-hot column (e.g. avoid dummy trap).

    Returns:
        Float tensor of one-hot columns with zero-variance columns removed.

    Note:
        Large cardinalities are expensive; consider ``torch.cuda.empty_cache()`` after neutralization.
    """
    if not isinstance(x , Tensor): 
        x = torch.FloatTensor(x)
    xmax = x.nan_to_num().max().int().item() 
    dummy = x.nan_to_num(xmax + 1).to(torch.int64)
    # dummy = torch.where(x.isnan() , m + 1 , x).to(torch.int64)
    dummy = F.one_hot(dummy).to(torch.float)[...,:xmax+1-ex_last] # slightly faster but will take a huge amount of memory
    dummy = dummy[...,dummy.sum(dim = tuple(range(x.dim()))) != 0]
    return dummy

def concat_factors(*factors : Tensor , dim : int = -1 , device = None) -> Tensor:
    """Concatenate non-None tensors along ``dim``.

    Args:
        *factors: Variable-length tensor arguments.
        dim: Concatenation dimension.
        device: If set, move result to this device.

    Returns:
        ``torch.concat`` of provided tensors.
    """
    facs = [f for f in factors if f is not None]
    ts = torch.concat(facs , dim = dim)
    if device is not None:
        ts = ts.to(device = device)
    return ts

def concat_factors_2d(*factors : Tensor , dim : int = -1 , device = None) -> Tensor:
    """Like ``concat_factors`` but unsqueezes any 2-D factor along ``dim`` first.

    Args:
        *factors: Tensors (2-D get an extra axis at ``dim``).
        dim: Concat dimension.
        device: Optional target device.

    Returns:
        Concatenated tensor.
    """
    facs = [f.unsqueeze(dim) if f.ndim == 2 else f for f in factors if f is not None]
    ts = torch.concat(facs , dim = dim)
    if device is not None:
        ts = ts.to(device = device)
    return ts

def neutralize_xdata_2d(x : Tensor | None , groups : None | list | tuple | Tensor = None):
    """Build design matrix: optional numeric ``x`` plus group dummies and intercept column.

    Args:
        x: Optional 2-D or 3-D regressors; expanded with dummies. If None and no groups, returns None.
        groups: Single tensor or iterable of group tensors for ``dummy`` one-hots.

    Returns:
        3-D float tensor ``[sample, features, ...]`` with leading constant column from ``F.pad``,
        or None when empty.

    Note:
        Memory-heavy when group cardinality is large.
    """
    if groups is None:
        groups = []
    elif not isinstance(groups , (list , tuple)): 
        groups  = [groups]
    if not groups and x is None:
        return None
    n_sample = x.shape[0] if x is not None else groups[0].shape[0]
    if x is None:
        x = torch.Tensor([]).reshape(n_sample,0,0).to(groups[0].device)
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    for g in groups: 
        x = concat_factors(x , dummy(g , ex_last=True))
    x.nan_to_num_(nan , nan , nan)
    x = F.pad(x , (1,0) , value = 1.)
    return x

def betas_torch(x : Tensor , y : Tensor , * , method = ['lstsq', 'inv']) -> Tensor:
    """Least-squares coefficients ``b`` minimizing ``||y - x @ b||`` (no intercept).

    Args:
        x: Design matrix ``(n, k)``.
        y: Targets ``(n, m)`` or ``(n, 1)``.
        method: Ordered list of fallbacks: ``'lstsq'`` then optional ``'inv'`` (regularized normal eqs).

    Returns:
        Coefficient tensor on success; on total failure, zeros ``(k, 1)`` on ``x``'s device/dtype.
    """
    x , y = x.float() , y.float()
    assert x.shape[0] == y.shape[0] , (x.shape , y.shape)
    b = None
    for m in method:
        try:
            if m == 'lstsq':
                b = torch.linalg.lstsq(x , y , rcond=None)[0]
            elif m == 'inv':
                M = x.T.mm(x)
                norm = M.norm()
                M_ = M / norm + torch.diag(torch.ones(len(M)) * 2e-5).to(x)
                b = (torch.linalg.inv(M_) / norm).mm(x.T).mm(y)
            else:
                raise ValueError(f'Invalid method: {m}')
            return b
        except Exception as e: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
            alert_message(f'neutralization error in betas_torch.{m}: {e}')

    return torch.zeros(x.shape[-1],1).to(x)

def betas_np(x : np.ndarray , y : np.ndarray , * , method = ['sk', 'lstsq', 'inv']) -> np.ndarray:
    """NumPy/sklearn OLS coefficients with method fallback chain.

    Args:
        x: ``(n, k)`` design.
        y: ``(n,)`` or ``(n, m)`` targets.
        method: Try ``'sk'`` (sklearn), ``'lstsq'``, then ``'inv'``.

    Returns:
        Coefficient array or ``zeros((k,1))`` if all methods fail.

    Raises:
        AssertionError: Row count mismatch.
    """
    assert x.shape[0] == y.shape[0] , (x.shape , y.shape)
    for m in method:
        try:
            if m == 'sk':
                b = LinearRegression(fit_intercept=False).fit(x, y).coef_.T
            elif m == 'lstsq':
                b = np.linalg.lstsq(x , y , rcond=None)[0]
            elif m == 'inv':
                M = x.T.dot(x)
                norm = np.linalg.norm(M)
                M_ = M / norm + np.diag(np.ones(len(M)) * 2e-5)
                b = (np.linalg.inv(M_) / norm) @ x.T @ y
            else:
                raise ValueError(f'Invalid method: {m}')
            return b
        except Exception as e: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
            alert_message(f'neutralization error in betas_np.{m}: {e}')
    return np.zeros((x.shape[-1],1))

def beta_calculator(method = 'np'):
    """Factory returning ``(x, y) -> betas`` using torch or NumPy backend.

    Args:
        method: ``'np'`` or ``'torch'``.

    Returns:
        Callable that dispatches to ``betas_np`` or ``betas_torch``.

    Raises:
        AssertionError: Invalid ``method``.
    """
    assert method in ['torch' , 'np'] , method
    def wrapper(x : Tensor | np.ndarray , y : Tensor | np.ndarray , **kwargs):
        """Dispatch to ``betas_np`` or ``betas_torch`` per factory ``method``."""
        if method == 'np':
            assert isinstance(x , np.ndarray) and isinstance(y , np.ndarray) , (type(x) , type(y))
            return betas_np(x , y)
        elif method == 'torch':
            assert isinstance(x , Tensor) and isinstance(y , Tensor) , (type(x) , type(y))
            return betas_torch(x , y)
    return wrapper

def neutralize_2d(y : Tensor | None , x : Tensor | None , * ,
                  dim : int = 0 , method = 'np' , zscore = True , device = None , inplace = False , 
                  min_coverage = 3):
    """Regress ``y`` on ``x`` slice-wise (cross-section or time via ``dim``) and subtract fit.

    Args:
        y: 2-D tensor to orthogonalize.
        x: 2-D or 3-D regressors (last dim = factors); expanded to 3-D internally.
        dim: ``0`` or ``1`` — which axis is treated as the "date" axis for row-wise LS loops.
        method: ``'np'`` or ``'torch'`` for beta solver.
        zscore: If True, apply ``zscore_inplace`` on result along ``dim``.
        device: Optional compute device for ``x,y`` during regression.
        inplace: If True, may mutate ``y`` in place before expand.
        min_coverage: Minimum finite rows per slice to estimate betas.

    Returns:
        Residual ``y`` on ``old_device``, same shape as input ``y``, or original ``y`` if skipped.

    Raises:
        AssertionError: Invalid shapes or ``method``/``dim``.
    """
    if x is None or y is None or x.numel() == 0: 
        return y
    assert method in ['np' , 'torch'] , method
    assert dim in [0,1] , dim
    assert y.ndim == 2 , y.dim()
    assert x.ndim in [2,3] , x.dim()
    
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    old_device = y.device

    valid_date_secid = torch.isfinite(x.sum(dim=-1)) & torch.isfinite(y) 

    x = x.nan_to_num(0,0,0)
    if inplace:
        y.nan_to_num_(nan , nan , nan).unsqueeze_(-1)
    else:
        y = y.nan_to_num(nan , nan , nan).unsqueeze(-1)
    if dim == 0: 
        # put date dimension to the first dimension
        x , y , valid_date_secid = x.permute(1,0,2) , y.permute(1,0,2) , valid_date_secid.permute(1,0)
    valid_feature = ~(x == 0).all(0) # [secid , factor]
    if device is not None:  
        x , y = x.to(device) , y.to(device)
    res = None
    if False and method == 'torch' and valid_date_secid.all() and valid_feature.all():
        # fastest, but cannot deal nan's which is always the case, so will not enter here
        try:
            model = torch.linalg.lstsq(x , y , rcond=None)
            res = (y - x @ model[0])
        except Exception:
            res = None
    if res is not None:
        y = res
    else: 
        betas_func = beta_calculator(method)
        if method == 'np': 
            xs , ys , dss , fs = x.cpu().numpy() , y.cpu().numpy() , valid_date_secid.cpu().numpy()  , valid_feature.cpu().numpy() 
        else:
            xs , ys , dss , fs = x , y , valid_date_secid , valid_feature
        for i , (y_ , x_ , s_ , k_) in enumerate(zip(ys , xs , dss , fs)):
            if s_.sum() < min_coverage or k_.sum() == 0: 
                continue
            betas = betas_func(x_[s_][:,k_] , y_[s_])
            ys[i,s_] -= (x_[s_][:,k_] @ betas) # type: ignore
        y = torch.FloatTensor(ys)
    if dim == 0: 
        y = y.permute(1,0,2)
    y = y.squeeze(-1).to(old_device)
    if zscore:
        y = zscore_inplace(y , dim = dim)
    return y

def neutralize_1d(y : Tensor | None , x : Tensor | None , insample : Tensor | None , * , 
                  method = 'torch' , zscore = True , device = None , inplace = False , 
                  min_coverage = 3):
    """Single cross-section neutralization: OLS of ``y`` on ``x`` over ``insample`` mask.

    Args:
        y: 1-D tensor.
        x: 1-D or 2-D regressors (column-expanded if 1-D).
        insample: Boolean mask same shape as ``y``; defaults to all True.
        method: ``'np'`` or ``'torch'``.
        zscore: Z-score residuals along dim 0 if True.
        device: Optional compute device.
        inplace: Allow in-place nan_to_num on ``y``.
        min_coverage: Minimum finite observations required.

    Returns:
        1-D residual vector on original device, or ``y`` if skipped.

    Raises:
        AssertionError: Shape/dim/method validation failures.
    """
    if x is None or y is None or x.numel() == 0: 
        return y

    if insample is None:
        insample = torch.ones_like(y).to(torch.bool)

    assert method in ['np' , 'torch'] , method
    assert y.shape == insample.shape , (y.shape , insample.shape)
    assert y.dim() == 1 , y.dim()
    
    if x.dim() == 1: 
        x = x.reshape(-1,1)
    old_device = y.device
    valid_x = torch.isfinite(x.sum(dim=-1)) & torch.isfinite(y)
    if valid_x.sum() < min_coverage: 
        return y

    x = x.nan_to_num(0,0,0)
    if inplace:
        y.nan_to_num_(nan , nan , nan).unsqueeze_(-1)
    else:
        y = y.nan_to_num(nan , nan , nan).unsqueeze(-1)
    
    if device is not None:  
        x , y = x.to(device) , y.to(device)
    betas_func = beta_calculator(method)
    if method == 'np': 
        xs , ys , ins = x.cpu().numpy() , y.cpu().numpy() , (valid_x * insample).cpu().numpy()
    else:
        xs , ys , ins = x , y , valid_x * insample

    x_ , y_ , = xs[ins] , ys[ins]
    k_ = ~(x_ == 0).all(0)
    betas = betas_func(x_[:,k_] , y_)
    ys[ins] = ys[ins] - (x_[:,k_] @ betas) # type: ignore
    ys = torch.FloatTensor(ys).squeeze_(-1).to(old_device)
    if zscore:
        ys = zscore_inplace(ys , dim = 0)
    return ys

def corrwith(x : Tensor , y : Tensor , * , dim : int | None = 1):
    """Pearson correlation along ``dim`` (NaN-aware, denominator floored).

    Args:
        x: Tensor, aligned with ``y``.
        y: Tensor, same shape as ``x``.
        dim: Reduced axis for means and correlation.

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

def covariance(x : Tensor , y : Tensor , * , dim : int | None = 1):
    """Sample covariance along ``dim`` (nan-mean removed).

    Args:
        x: Tensor.
        y: Tensor, aligned with ``x``.
        dim: Axis for demeaning and reduction.

    Returns:
        Covariance with ``dim`` reduced.
    """
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  # [TS, C]
    y_ymean = y - torch.nanmean(y, dim, keepdim=True)  # [TS, C]
    cov = torch.nansum(x_xmean * y_ymean, dim)  # [TS, 1]
    return cov

def beta(x : Tensor , y : Tensor , * , dim : int | None = 1):
    """OLS slope ``cov(x,y) / var(x)`` along ``dim``.

    Args:
        x: Regressor tensor.
        y: Dependent tensor.
        dim: Reduction axis.

    Returns:
        Beta tensor.
    """
    return covariance(x,y,dim=dim) / covariance(x,x,dim=dim)

def beta_pos(x : Tensor , y : Tensor , * , dim : int | None = 1):
    """Beta using only observations with ``x >= 0`` (others set NaN).

    Args:
        x: Regressor.
        y: Dependent.
        dim: Reduction axis.

    Returns:
        Restricted beta tensor.
    """
    y = torch.where(x < 0 , nan , y)
    x = torch.where(x < 0 , nan , x)
    return covariance(x,y,dim=dim) / covariance(x,x,dim=dim)

def beta_neg(x : Tensor , y : Tensor , * , dim : int | None = 1):
    """Beta using only observations with ``x <= 0`` (others set NaN).

    Args:
        x: Regressor.
        y: Dependent.
        dim: Reduction axis.

    Returns:
        Restricted beta tensor.
    """
    y = torch.where(x > 0 , nan , y)
    x = torch.where(x > 0 , nan , x)
    return covariance(x,y,dim=dim) / covariance(x,x,dim=dim)

def stddev(x : Tensor , * , dim : int | None = 1):
    """Root sum of squared deviations from nan-mean along ``dim`` (not normalized by n).

    Args:
        x: Input tensor.
        dim: Axis for mean and sum of squares.

    Returns:
        ``sqrt(nansum((x - mean)^2))`` along ``dim``.
    """
    x_xmean  = x - torch.nanmean(x , dim, keepdim=True)
    return torch.nansum(x_xmean.square(), dim).sqrt()

def zscore(x : Tensor , * , dim : int | None = 0 , index : int | None = None):
    """Z-score along ``dim``, optionally selecting one slice after scaling.

    Args:
        x: Input tensor.
        dim: Axis for mean and std.
        index: If not None, ``select(dim, index)`` on the centered tensor before dividing.

    Returns:
        Z-scores with adaptive epsilon from mean absolute std.
    """
    x_xmean  = x - torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = torch.nansum(x_xmean.square(), dim , keepdim=index is None).sqrt()
    if index is not None: 
        x_xmean = x_xmean.select(dim,index)
    z = x_xmean / (x_stddev + 1e-4 * x_stddev.nanmean())
    return z

def abs(x):
    """Elementwise ``torch.abs``.

    Args:
        x: Input tensor.

    Returns:
        Absolute values.
    """
    return torch.abs(x)

def zscore_inplace(x : Tensor , * , dim : int | None = 0):
    """In-place z-score: subtract nan-mean, divide by nanstd with epsilon (mutates ``x``).

    Args:
        x: Tensor modified in place.
        dim: Statistics axis.

    Returns:
        Same tensor ``x`` after scaling.
    """
    x -= torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = nanstd(x , dim = dim , keepdim = True)
    x /= (x_stddev + 1e-4 * x_stddev.abs().nanmean())
    return x

def add(x : Tensor , y : Tensor):
    """Elementwise sum ``x + y``.

    Args:
        x: Tensor.
        y: Tensor broadcastable to ``x``.

    Returns:
        Sum tensor.
    """
    return x + y

def sub(x : Tensor , y : Tensor):
    """Elementwise difference ``x - y``.

    Args:
        x: Tensor.
        y: Tensor broadcastable to ``x``.

    Returns:
        Difference tensor.
    """
    return x - y

def mul(x : Tensor , y : Tensor):
    """Elementwise product ``x * y``.

    Args:
        x: Tensor.
        y: Tensor broadcastable to ``x``.

    Returns:
        Product tensor.
    """
    return x * y

def div(x : Tensor , y : Tensor):
    """Elementwise quotient ``x / y``.

    Args:
        x: Tensor.
        y: Tensor broadcastable to ``x``.

    Returns:
        Quotient tensor.
    """
    return x / y

def add_int(x : Tensor , d : int):
    """Add scalar ``d`` to tensor ``x``.

    Args:
        x: Tensor.
        d: Integer offset.

    Returns:
        ``x + d``.
    """
    return x + d

def sub_int1(x : Tensor , d : int):
    """Subtract scalar ``d`` from ``x`` (variant 1).

    Args:
        x: Tensor.
        d: Integer subtrahend.

    Returns:
        ``x - d``.
    """
    return x - d

def sub_int2(x : Tensor , d : int):
    """Subtract scalar ``d`` from ``x`` (variant 2, same as ``sub_int1``).

    Args:
        x: Tensor.
        d: Integer subtrahend.

    Returns:
        ``x - d``.
    """
    return x - d

def mul_int(x : Tensor , d : int):
    """Multiply ``x`` by integer ``d``.

    Args:
        x: Tensor.
        d: Integer multiplier.

    Returns:
        ``x * d``.
    """
    return x * d

def div_int1(x : Tensor , d : int):
    """Divide ``x`` by integer ``d`` (variant 1).

    Args:
        x: Tensor.
        d: Integer divisor.

    Returns:
        ``x / d``.
    """
    return x / d

def div_int2(x : Tensor , d : int):
    """Divide ``x`` by integer ``d`` (variant 2).

    Args:
        x: Tensor.
        d: Integer divisor.

    Returns:
        ``x / d``.
    """
    return x / d

def neg(x : Tensor):
    """Negate tensor.

    Args:
        x: Input.

    Returns:
        ``-x``.
    """
    return -x

def neg_int(x : int):
    """Negate integer.

    Args:
        x: Integer.

    Returns:
        ``-x``.
    """
    return -x

def sigmoid(x : Tensor):
    """Logistic sigmoid ``1 / (1 + exp(-x))``.

    Args:
        x: Input tensor.

    Returns:
        Sigmoid tensor.
    """
    return 1 / (1 + torch.exp(-x))

def rank_sub(x : Tensor , y : Tensor , * , dim : int | None = 0):
    """Difference of rank percentiles.

    Args:
        x: First operand.
        y: Second operand.
        dim: Axis for ``rank_pct``.

    Returns:
        ``rank_pct(x) - rank_pct(y)``.
    """
    return rank_pct(x,dim=dim) - rank_pct(y,dim=dim)

def rank_add(x : Tensor , y : Tensor , * , dim : int | None = 0):
    """Sum of rank percentiles.

    Args:
        x: First operand.
        y: Second operand.
        dim: Axis for ``rank_pct``.

    Returns:
        ``rank_pct(x) + rank_pct(y)``.
    """
    return rank_pct(x,dim=dim) + rank_pct(y,dim=dim)

def rank_div(x : Tensor , y : Tensor , * , dim : int | None = 0):
    """Ratio of rank percentiles.

    Args:
        x: Numerator.
        y: Denominator.
        dim: Axis for ``rank_pct``.

    Returns:
        ``rank_pct(x) / rank_pct(y)``.
    """
    return rank_pct(x,dim=dim) / rank_pct(y,dim=dim)

def rank_mul(x : Tensor , y : Tensor , * , dim : int | None = 0):
    """Product of rank percentiles.

    Args:
        x: First operand.
        y: Second operand.
        dim: Axis for ``rank_pct``.

    Returns:
        ``rank_pct(x) * rank_pct(y)``.
    """
    return rank_pct(x,dim=dim) * rank_pct(y,dim=dim)

def log(x : Tensor):
    """Elementwise natural logarithm.

    Args:
        x: Tensor.

    Returns:
        ``x.log()``.
    """
    return x.log()

def sqrt(x : Tensor):
    """Elementwise square root.

    Args:
        x: Tensor.

    Returns:
        ``x.sqrt()``.
    """
    return x.sqrt()

def square(x : Tensor):
    """Elementwise square.

    Args:
        x: Tensor.

    Returns:
        ``x.square()``.
    """
    return x.square()

def lin_decay(x : Tensor , * , dim = 1):
    """Linearly weighted nan-sum along ``dim`` (weights ``1..L`` on the rolling axis).

    Args:
        x: Tensor (typical rolling window layout).
        dim: Axis carrying weights ``1, 2, ..., size[dim]``.

    Returns:
        Weighted sum divided by sum of weights on finite entries.
    """
    if dim < 0:
        dim = len(x.shape) + dim
    raw_shape = x.shape
    weight_shape = [1 if i != dim else raw_shape[dim] for i in range(len(raw_shape))]
    weight = torch.arange(1, raw_shape[dim] + 1, 1).reshape(weight_shape).to(x)
    return (x * weight).nansum(dim=dim) / ((weight * (~x.isnan())).sum(dim=dim))

def sign(x : Tensor):
    """Elementwise sign.

    Args:
        x: Tensor.

    Returns:
        ``x.sign()``.
    """
    return x.sign()

def scale(x : Tensor , c = 1 , * , dim = 0):
    """Scale ``x`` by ``c`` divided by L1 norm along ``dim`` (nan-aware sum of abs).

    Args:
        x: Tensor.
        c: Scalar scale (default 1).
        dim: Axis for normalization denominator.

    Returns:
        Normalized tensor times ``c``.
    """
    return c * x / x.abs().nansum(dim=dim, keepdim=True)

def signedpower(x : Tensor , a : float):
    """``sign(x) * |x|^a`` elementwise.

    Args:
        x: Tensor.
        a: Exponent.

    Returns:
        Signed power tensor.
    """
    return x.sign() * x.abs().pow(a)

def pctchg(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Relative change vs ``d``-lag along ``dim``: ``(x - x_lag) / |x_lag|``.

    Args:
        x: Price or level tensor.
        d: Lag length (positive = look back).
        dim: Lag axis.

    Returns:
        Fractional change tensor.
    """
    return (x - ts_delay(x,d , dim = dim)) / abs(ts_delay(x,d , dim = dim))

def ts_delay(x : Tensor , d : int , * , dim : Literal[1] = 1, no_alert = False):
    """Circular roll along ``dim`` with leading segment zeroed (NaN-filled) to avoid wrap.

    Args:
        x: Input tensor.
        d: Shift amount; positive shifts "forward" in index (past values move to current).
        dim: Axis to roll.
        no_alert: If False, warn on negative ``d`` (lookahead risk).

    Returns:
        Lagged tensor; all NaN if ``d`` exceeds axis length.

    Note:
        Boundary clearing uses ``z[:, :d]`` / ``z[:, d:]`` when ``dim==1``; other ``dim`` values
        still roll correctly but the cleared slab pattern matches the historical dim-1 convention.
    """
    if d > x.shape[dim]: 
        return x * nan
    if d < 0 and not no_alert: 
        alert_message('Beware! future information used!' , color = 'lightred')
    z = x.roll(d, dims=dim)
    if d >= 0:
        z[:,:d] = nan
    else:
        z[:,d:] = nan
    return z

def ts_delta(x : Tensor , d : int , * , dim : Literal[1] = 1, no_alert = False):
    """``x - ts_delay(x, d)`` along ``dim``.

    Args:
        x: Input tensor.
        d: Lag for difference.
        dim: Axis (same as ``ts_delay``).
        no_alert: Passed to ``ts_delay`` for negative ``d`` warning.

    Returns:
        Difference tensor.

    Note:
        Uses ``x.shape[0]`` for oversize check (legacy); prefer consistent axis length checks for ``dim``.
    """
    if d > x.shape[0]: 
        return x * nan
    if d < 0 and not no_alert: 
        alert_message('Beware! future information used!' , color = 'lightred')
    z = x - ts_delay(x, d, dim=dim)
    return z

@TsRoller.decor(1)
def ts_zscore(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling ending value of ``zscore`` along the window (last window position).

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis (default 1).

    Returns:
        Rolling z-score tensor (one value per window end).
    """
    return zscore(x , dim = dim , index = -1)

@TsRoller.decor(1)
def ts_mean(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling nan-mean over windows of length ``d``.

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling mean tensor.
    """
    return torch.nanmean(x , dim = dim)

@TsRoller.decor(1 , nan = np.inf)
def ts_min(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling minimum over each window.

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling minima.
    """
    return torch.min(x , dim=dim).values

@TsRoller.decor(1 , nan = -np.inf)
def ts_max(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling maximum over each window.

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling maxima.
    """
    return torch.max(x , dim=dim).values

@TsRoller.decor(1 , nan = np.inf)
def ts_argmin(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling argmin (as float) over each window.

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Indices of minima per window.
    """
    return torch.argmin(x , dim=dim).to(torch.float)

@TsRoller.decor(1 , nan = -np.inf)
def ts_argmax(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling argmax (as float) over each window.

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Indices of maxima per window.
    """
    return torch.argmax(x , dim=dim).to(torch.float)

@TsRoller.decor(1)
def ts_rank(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling ending rank percentile (last element of window-wise ``rank_pct``).

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rank percentile at window end.
    """
    return rank_pct(x,dim=dim)[...,-1]

@TsRoller.decor(1 , nan = 0)
def ts_stddev(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling standard deviation over each window.

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling ``torch.std`` along the window axis.
    """
    return torch.std(x,dim=dim)

@TsRoller.decor(1 , nan = 0)
def ts_sum(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling sum over each window.

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling sums.
    """
    return torch.sum(x,dim=dim)

@TsRoller.decor(1 , nan = 1)
def ts_product(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling product over each window.

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling products.
    """
    return torch.prod(x,dim=dim)

@TsRoller.decor(1)
def ts_lin_decay(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling linear-decay weighted average (see ``lin_decay``).

    Args:
        x: Input tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling weighted average tensor.
    """
    return lin_decay(x , dim=dim)

def ts_decay_pos_dif(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling linear-decay of ``relu(x - y)``.

    Args:
        x: Minuend tensor.
        y: Subtrahend tensor.
        d: Window length.
        dim: Rolling axis.

    Returns:
        ``ts_lin_decay`` applied to the positive part of ``x - y``.
    """
    value = x - y
    value = value.clip(0)
    return ts_lin_decay(value, d , dim=dim)

@TsRoller.decor(2,nan=0)
def ts_corr(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling Pearson correlation between ``x`` and ``y`` over each window.

    Args:
        x: First series.
        y: Second series, aligned with ``x``.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling correlations.
    """
    return corrwith(x , y , dim=dim)

@TsRoller.decor(2 , nan = 0)
def ts_beta(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling OLS beta ``cov(x,y)/var(x)`` inside each window.

    Args:
        x: Regressor.
        y: Dependent.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling betas.
    """
    return beta(x , y , dim=dim)

@TsRoller.decor(2 , nan = 0)
def ts_beta_pos(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling beta using only ``x >= 0`` observations within each window.

    Args:
        x: Regressor.
        y: Dependent.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling restricted betas.
    """
    return beta_pos(x , y , dim=dim)

@TsRoller.decor(2 , nan = 0)
def ts_beta_neg(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling beta using only ``x <= 0`` observations within each window.

    Args:
        x: Regressor.
        y: Dependent.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling restricted betas.
    """
    return beta_neg(x , y , dim=dim)

@TsRoller.decor(2 , nan = 0)
def ts_cov(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling covariance between ``x`` and ``y``.

    Args:
        x: First series.
        y: Second series.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling covariances.
    """
    return covariance(x , y , dim=dim)

@TsRoller.decor(2,nan=0)
def ts_rankcorr(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """Rolling Spearman-style correlation using ``rank_pct`` of ``x`` and ``y``.

    Args:
        x: First series.
        y: Second series.
        d: Window length.
        dim: Rolling axis.

    Returns:
        Rolling rank correlations.
    """
    return corrwith(rank_pct(x,dim=dim) , rank_pct(y,dim=dim) , dim=dim)

@TsRoller.unfold_chunk_slice_x()
def conditional_x(
    x : Tensor , d : int , n : int , method : Literal['btm' , 'top' , 'diff'] , * ,
    dim : Literal[1] = 1, use : Literal['mean' , 'thres'] = 'mean',
    force_directional_sign : bool = False
):
    """Rolling statistic of ``x`` conditioned on ``x`` being among the smallest/largest ``n`` in the window.

    Args:
        x: Input tensor.
        d: Window length.
        n: How many smallest/largest points to select (capped by ``d``).
        method: ``'btm'`` (small ``x``), ``'top'`` (large ``x``), or ``'diff'`` (top minus bottom groups).
        dim: Rolling axis.
        use: ``'mean'`` of selected values or ``'thres'``-style max of masked tensor.
        force_directional_sign: Clip thresholds for directional filters (see implementation).

    Returns:
        Folded rolling tensor of conditional means or their difference.

    Raises:
        AssertionError: Invalid ``method``.
        ValueError: If neither branch produces a group.
    """
    assert method in ['btm' , 'top' , 'diff'] , method
    n = min(d, n)
    x = TsRoller.unfold(x , d , dim = dim)
    groups : list[Tensor | None] = [None , None]
    if method in ['btm' , 'diff']:
        condition = kthvalue_by_topk(x, n, dim=-1, keepdim=True, largest=False)
        if force_directional_sign: 
            condition = condition.clip(max=0)
        value = torch.where(x <= condition , x , nan).nan_to_num_(nan , nan , nan)
        groups[0] = value.nanmean(dim=-1) if use == 'mean' else value.nan_to_num(-torch.inf).max(dim=-1).values
    if method in ['top' , 'diff']:
        condition = kthvalue_by_topk(x, n, dim=-1, keepdim=True, largest=True)
        if force_directional_sign: 
            condition = condition.clip(min=0)
        value = torch.where(x >= condition , x , nan).nan_to_num_(nan , nan , nan)
        groups[1] = value.nanmean(dim=-1) if use == 'mean' else value.nan_to_num(-torch.inf).max(dim=-1).values
    
    z = [grp for grp in groups if grp is not None]
    if len(z) == 1:
        z = z[0]
    elif len(z) == 2:
        z = z[1] - z[0]
    else:
        raise ValueError(f'Invalid number of groups: {len(z)}')
    z = TsRoller.fold(z , d , dim = dim , nan = nan)
    return z

@TsRoller.unfold_chunk_slice_xy()
def conditional_y_on_x(
    x : Tensor , y : Tensor , d : int , n : int , method : Literal['btm' , 'top' , 'diff'] , * ,
    dim : Literal[1] = 1, use : Literal['mean' , 'thres'] = 'mean',
    force_directional_sign : bool = False
):
    """Rolling statistic of ``y`` where ``x`` selects the bottom/top ``n`` observations in each window.

    Args:
        x: Conditioning series.
        y: Target series (aligned with ``x``).
        d: Window length.
        n: Count of extreme ``x`` values to average over.
        method: ``'btm'``, ``'top'``, or ``'diff'`` (spread between top and bottom groups on ``y``).
        dim: Rolling axis.
        use: ``'mean'`` or ``'thres'`` aggregation of masked ``y``.
        force_directional_sign: Threshold clipping for signed ``x`` filters.

    Returns:
        Folded rolling conditional expectation tensor.

    Raises:
        AssertionError: Invalid ``method``.
        ValueError: If group list is empty.
    """
    assert method in ['btm' , 'top' , 'diff'] , method
    n = min(d, n)
    x = TsRoller.unfold(x , d , dim = dim)
    y = TsRoller.unfold(y , d , dim = dim)
    groups : list[Tensor | None] = [None , None]
    if method in ['btm' , 'diff']:
        condition = kthvalue_by_topk(x, n, dim=-1, keepdim=True, largest=False)
        if force_directional_sign: 
            condition = condition.clip(max=0)
        value = torch.where(x <= condition , y , nan).nan_to_num_(nan , nan , nan)
        groups[0] = value.nanmean(dim=-1) if use == 'mean' else value.nan_to_num(-torch.inf).max(dim=-1).values
        
    if method in ['top' , 'diff']:
        condition = kthvalue_by_topk(x, n, dim=-1, keepdim=True, largest=True)
        if force_directional_sign: 
            condition = condition.clip(min=0)
        value = torch.where(x >= condition , y , nan).nan_to_num_(nan , nan , nan)
        groups[1] = value.nanmean(dim=-1) if use == 'mean' else value.nan_to_num(-torch.inf).max(dim=-1).values
        
    z = [grp for grp in groups if grp is not None]
    if len(z) == 1:
        z = z[0]
    elif len(z) == 2:
        z = z[1] - z[0]
    else:
        raise ValueError(f'Invalid number of groups: {len(z)}')
    z = TsRoller.fold(z , d , dim = dim , nan = nan)
    return z

def ts_btm_y_on_x(x : Tensor , y : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    """Mean of ``y`` on the ``n`` smallest ``x`` within each length-``d`` window.

    Args:
        x: Conditioning series.
        y: Target series.
        d: Window length.
        n: Bottom count.
        dim: Rolling axis.

    Returns:
        ``conditional_y_on_x`` with ``method='btm'``.
    """
    return conditional_y_on_x(x, y, d, n, dim=dim, method='btm')

def ts_top_y_on_x(x : Tensor , y : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    """Mean of ``y`` on the ``n`` largest ``x`` within each window.

    Args:
        x: Conditioning series.
        y: Target series.
        d: Window length.
        n: Top count.
        dim: Rolling axis.

    Returns:
        ``conditional_y_on_x`` with ``method='top'``.
    """
    return conditional_y_on_x(x, y, d, n, dim=dim, method='top')

def ts_dif_y_on_x(x : Tensor , y : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    """Top-``n`` minus bottom-``n`` mean of ``y`` ordered by ``x`` within each window.

    Args:
        x: Conditioning series.
        y: Target series.
        d: Window length.
        n: Tail count on each side.
        dim: Rolling axis.

    Returns:
        ``conditional_y_on_x`` with ``method='diff'``.
    """
    return conditional_y_on_x(x, y, d, n, dim=dim, method='diff')

def ts_btm_x(x : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    """Mean of smallest ``n`` values of ``x`` in each rolling window.

    Args:
        x: Input series.
        d: Window length.
        n: Bottom count.
        dim: Rolling axis.

    Returns:
        ``conditional_x`` with ``method='btm'``.
    """
    return conditional_x(x, d, n, dim=dim, method='btm')

def ts_top_x(x : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    """Mean of largest ``n`` values of ``x`` in each rolling window.

    Args:
        x: Input series.
        d: Window length.
        n: Top count.
        dim: Rolling axis.

    Returns:
        ``conditional_x`` with ``method='top'``.
    """
    return conditional_x(x, d, n, dim=dim, method='top')

def ts_dif_x(x : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    """Difference between top-``n`` and bottom-``n`` means of ``x`` in each window.

    Args:
        x: Input series.
        d: Window length.
        n: Tail count on each side.
        dim: Rolling axis.

    Returns:
        ``conditional_x`` with ``method='diff'``.
    """
    return conditional_x(x, d, n, dim=dim, method='diff')

def transpose_qkv(X : torch.Tensor , num_heads : int):
    """Reshape projections for multi-head attention: merge batch*heads.

    Args:
        X: Tensor shaped ``(batch, seq, num_heads * head_dim)`` or compatible first three dims.
        num_heads: Number of attention heads.

    Returns:
        Tensor shaped ``(batch * num_heads, seq, head_dim)``.
    """
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X : torch.Tensor , num_heads : int):
    """Inverse of ``transpose_qkv``: merge head dimension back into last feature dim.

    Args:
        X: Tensor shaped ``(batch * num_heads, seq, head_dim)``.
        num_heads: Head count used when splitting.

    Returns:
        Tensor shaped ``(batch, seq, num_heads * head_dim)``.
    """
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)