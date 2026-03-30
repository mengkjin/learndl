"""Pandas/NumPy/statsmodels helpers: ranking, z-score, winsorization, regression, and label utilities."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

from typing import Any , Literal
from scipy.stats import norm, rankdata
from scipy.linalg import lstsq

from .basic import alert_message , DIV_TOL

warnings.filterwarnings('ignore' , category=RuntimeWarning , message='Mean of empty slice')

def fill_na_as_const(x_: np.ndarray, c_ : float = 0.0):
    """Replace NaNs with a constant (copy).

    Args:
        x_: Input array.
        c_: Fill value for NaN positions.

    Returns:
        New array with NaNs replaced by ``c_``.
    """
    rtn = x_.copy()
    rtn[np.isnan(x_)] = c_
    return rtn

def multi_bin_label(x : np.ndarray , n = 10):
    """Quantile-based multi-bin labels and nonnegative weights.

    Args:
        x: 1-D array.
        n: Number of equal-frequency bins.

    Returns:
        Tuple ``(y, w)`` where ``y`` are bin scores ``2*i - n + 1`` and ``w = abs(y)``.
    """
    y , w = np.zeros_like(x) , np.zeros_like(x)
    for i in range(n):
        low , high = np.quantile(x, i/n) , np.quantile(x, (i+1)/n)
        if i == n-1:
            y[(x >= low)] = 2 * i - n + 1
        elif i == 0:
            y[(x < high)] = 2 * i - n + 1
        else:
            y[(x >= low) & (x < high)] = 2 * i - n + 1
    w[:] = np.abs(y)
    return y, w

def bin_label(x : np.ndarray):
    """Binary label at median split with simple weights.

    Args:
        x: 1-D array.

    Returns:
        Tuple ``(y, w)`` with ``y`` in ``{0,1}`` above/below ``nanmedian`` and ``w = y + 1``.
    """
    y , w = np.zeros_like(x) , np.zeros_like(x)
    y[x >= np.nanmedian(x)] = 1
    w[:] = y + 1
    return y, w

def norm_to_1(x_: np.ndarray , value : float = 1.):
    """Min-max scale to ``[0, value]`` ignoring NaNs.

    Args:
        x_: Input array.
        value: Target max after scaling.

    Returns:
        Scaled array, or zeros if empty or constant input.
    """
    if len(x_) and value:
        if np.nanmax(x_) - np.nanmin(x_) == 0:
            return np.zeros_like(x_)
        else:
            return (x_ - np.nanmin(x_)) / (np.nanmax(x_) - np.nanmin(x_)) * value
    else:
        return np.zeros_like(x_)

def rank(x_: np.ndarray, is_zscore=True, is_norm_inv=False):
    """Dense rank with optional z-score or normal-inverse transform.

    Args:
        x_: Input; NaNs imputed with ``nanmedian`` for ranking when possible.
        is_zscore: If True and not ``is_norm_inv``, z-score the dense ranks.
        is_norm_inv: If True, map ranks through normal PPF and re-standardize.

    Returns:
        Transformed 1-D array, or all-NaN if ranking fails after imputation.
    """
    x = x_.copy()
    nan_flg = np.isnan(x)
    if nan_flg.any():
        x[nan_flg] = np.nanmedian(x_)
    if not np.isnan(x).any():
        rtn = rankdata(x, method='dense')
        if is_zscore and not is_norm_inv:
            rtn = (rtn - rtn.mean()) / rtn.std()
        if is_norm_inv:
            rtn = rtn / (1 + np.max(rtn))
            rtn = norm.ppf(rtn)
            rtn_std = rtn.std()
            if rtn_std <= 0.0:
                rtn = np.zeros_like(rtn)  # 0.0 is the mean, mode, median of a norm distribution
            else:
                rtn = rtn / rtn_std
    else:
        rtn = np.full_like(x, np.nan)
    return rtn

def rank_in_category(x_: list[np.ndarray], category_: np.ndarray):
    """Rank each array in ``x_`` within ``category_`` groups.

    Args:
        x_: List of 1-D arrays aligned with ``category_``.
        category_: Category labels, same length as each ``x_[i]``.

    Returns:
        List of arrays with group-wise ``rank(..., True, True)``; singleton groups get 0.0.
    """
    cate_list = list(np.unique(category_))
    rtn = list()
    for i in range(len(x_)):
        rtn.append(np.zeros_like(category_) * np.nan)
    for cate in cate_list:
        fl = category_ == cate
        for i in range(len(x_)):
            tmp = x_[i][fl]
            if tmp.shape[0] == 1:
                rtn[i][fl] = 0.0
            else:
                rtn[i][fl] = rank(tmp, True, True)
    return rtn

def norm_inv(x_: np.ndarray):
    """Apply ``scipy.stats.norm.ppf`` elementwise (open interval (0,1)).

    Args:
        x_: Strictly in ``(0, 1)``; empty array returns zeros.

    Returns:
        Normal inverse-CDF values.

    Raises:
        AssertionError: If any value is outside ``(0, 1)``.
    """
    if len(x_) == 0: 
        return x_ * 0.0
    assert (x_ > 0).all() , x_.min()
    assert (x_ < 1).all() , x_.max()
    rtn = norm.ppf(x_)
    return rtn

def zscore(x_: np.ndarray):
    """Cross-sectional z-score: subtract ``nanmean``, divide by ``nanstd + DIV_TOL``.

    Args:
        x_: 1-D array.

    Returns:
        Z-scored array.
    """
    u = np.nanmean(x_)
    d = np.nanstd(x_) + DIV_TOL
    return (x_ - u) / d

def winsorize_by_bnd(src_: np.ndarray, u_prc_: float = 0.95, l_prc_: float = 0.05):
    """Clip a 1-D series to percentile bounds.

    Args:
        src_: 1-D array.
        u_prc_: Upper percentile in ``(0, 100)``.
        l_prc_: Lower percentile, strictly less than ``u_prc_``.

    Returns:
        ``clip`` of ``src_`` to ``[l, u]`` percentiles.

    Raises:
        AssertionError: On invalid shape or percentile ordering.
    """
    assert src_.ndim == 1 , src_.shape
    assert 0 < l_prc_ and l_prc_ < u_prc_ and u_prc_ < 100 , (l_prc_ , u_prc_)
    ulb = np.nanpercentile(src_, [u_prc_, l_prc_])
    des = src_.clip(ulb[1], ulb[0])
    return des

def winsorize_by_dist(src_: np.ndarray, m_: float = 3.5, winsor_rng : float = 0. ,dist_type_: int=0):
    """Winsorize using mean ± ``m_`` standard deviations (normal or log-normal working space).

    Args:
        src_: 1-D array (strictly positive if ``dist_type_ == 1``).
        m_: Half-width in standard deviations.
        winsor_rng: Scale for soft extension past bounds via ``norm_to_1``.
        dist_type_: ``0`` normal, ``1`` log-normal (operate on ``log`` then ``exp``).

    Returns:
        Winsorized 1-D array.

    Raises:
        AssertionError: On invalid parameters or non-positive values in log mode.
    """
    assert src_.ndim == 1 , src_.shape
    assert m_ > 0 , m_
    assert winsor_rng > 0 , winsor_rng
    assert dist_type_ in (0, 1) , dist_type_
    if dist_type_ == 1:
        assert np.all(np.logical_or(np.isnan(src_), src_ > 0)), "winsorize_by_dist>>src_ must be positive when dist type as 1(log_normal)"
        des = np.log(src_)
    else:
        des = src_
    avg = np.nanmean(des)
    std = np.nanstd(des)
    ub = avg + m_ * std
    lb = avg - m_ * std
    des[des > ub] = ub + norm_to_1(des[des > ub] , winsor_rng)
    des[des < lb] = lb + norm_to_1(des[des < lb] , winsor_rng) - winsor_rng
    
    if dist_type_ == 1:
        des = np.exp(des)
    return des

def winsorize_by_bnded_dist(src_: np.ndarray, bnd_: float = 0.1, m_: float = 3.5, dist_type_: int=0):
    """Winsorize using robust stats on a central ``(bnd_, 100-bnd_)`` percentile band.

    Args:
        src_: 1-D array.
        bnd_: Percentile tail mass excluded from mean/std estimation (each side).
        m_: Multiplier for band half-width in SD units after clipping to in-band observations.
        dist_type_: ``0`` normal, ``1`` log-normal.

    Returns:
        Clipped 1-D array in working space, exponentiated if log-normal.

    Raises:
        AssertionError: Same family as ``winsorize_by_dist``.
    """
    assert bnd_ > 0 and bnd_ < 50 , bnd_
    assert src_.ndim == 1 , src_.shape
    assert m_ > 0 , m_
    assert dist_type_ in (0, 1) , dist_type_
    if dist_type_ == 1:
        assert np.all(np.logical_or(np.isnan(src_), src_ > 0)), "winsorize_by_dist>>src_ must be positive when dist type as 1(log_normal)"
        des = np.log(src_)
    else:
        des = src_
    lu = np.nanpercentile(des, [bnd_, 100 - bnd_])
    bnded_flg = np.logical_and(des >= lu[0], des <= lu[1])
    avg = np.nanmean(des[bnded_flg])
    std = np.nanstd(des[bnded_flg])
    ub = avg + m_ * std
    lb = avg - m_ * std
    des = des.clip(lb, ub)
    if dist_type_ == 1:
        des = np.exp(des)
    return des

def patched_by_industry(y_: np.ndarray, ind_risk_: np.ndarray, method_: int=0):
    """Fill NaNs in ``y_`` using industry-group median or mean.

    Args:
        y_: 1-D dependent values.
        ind_risk_: 1-D industry codes aligned with ``y_``.
        method_: ``0`` for ``nanmedian``, ``1`` for ``nanmean`` per industry.

    Returns:
        Copy of ``y_`` with NaNs replaced where possible.

    Raises:
        AssertionError: On shape mismatch or unresolved remaining NaNs after patch.
    """
    assert method_ in (0, 1) , method_
    assert y_.ndim == 1 and ind_risk_.ndim == 1 , (y_.shape , ind_risk_.shape)
    nan_flag = np.isnan(y_)
    if nan_flag.any():
        nan_ind = ind_risk_[nan_flag]
        nan_ind_list = np.unique(nan_ind)
        replaced_vals = np.zeros(len(nan_ind)) * np.nan
        for ind in nan_ind_list:
            if method_ == 0:
                replaced_vals[nan_ind == ind] = np.nanmedian(y_[ind_risk_ == ind])
            else:
                replaced_vals[nan_ind == ind] = np.nanmean(y_[ind_risk_ == ind])
        if np.isnan(replaced_vals).any():
            alert_message('warning >> patched_by_industry >> all nan encountered.')
            replaced_vals[np.isnan(replaced_vals)] = np.nanmedian(replaced_vals)
        assert not np.isnan(replaced_vals).any() , replaced_vals
        y = y_.copy()
        y[nan_flag] = replaced_vals
    else:
        y = y_.copy()
    return y

def trim(v , v1 , v2):
    """Set values below ``v1`` or above ``v2`` to NaN.

    Args:
        v: Numeric array (copied as float).
        v1: Lower bound or None to skip.
        v2: Upper bound or None to skip.

    Returns:
        Modified array.
    """
    v = v + 0.
    if v1 is not None: 
        v[v < v1] = np.nan
    if v2 is not None: 
        v[v > v2] = np.nan
    return v

def winsor(v , v1 , v2):
    """Clip ``v`` to ``[v1, v2]`` where bounds are not None.

    Args:
        v: Numeric array.
        v1: Lower clip or None.
        v2: Upper clip or None.

    Returns:
        Clipped array.
    """
    v = v + 0.
    if v1 is not None or v2 is not None:
        v = v.clip(v1, v2)
    return v

def weighted_mean(v , weight = None):
    """Global nan-mean, optionally weighted.

    Args:
        v: Array-like.
        weight: Same shape as ``v`` or None for uniform weights.

    Returns:
        Scalar weighted mean over all elements.
    """
    if weight is not None:
        weight = np.nan_to_num(weight)
        return np.nansum(v * weight , axis = None) / (np.nansum(weight , axis = None) + DIV_TOL)
    else:
        return np.nanmean(v , axis = None)

def whiten(v : pd.DataFrame | pd.Series | np.ndarray , weight = None) -> Any:
    """Subtract weighted mean and divide by global nan std (with ``DIV_TOL`` for non-array path).

    Args:
        v: Series, DataFrame, or ndarray.
        weight: Optional weights for the mean (see ``weighted_mean``).

    Returns:
        Same container type as ``v``, standardized.
    """
    stdev = np.nanstd(v) if isinstance(v , np.ndarray) else np.nanstd(v.to_numpy(float).flatten()) + DIV_TOL
    mean = weighted_mean(v , weight)
    return (v - mean) / stdev

def winsorize(v , 
              center : Literal['median' , 'mean'] = 'median', 
              scale : Literal['mad' , 'sd'] = 'mad', 
              const : float | np.floating | None = None , 
              trim_val : tuple[float | None,float | None] = (None , None) , 
              winsor_val : tuple[float | None,float | None] = (None , None) , 
              winsor_pct : tuple[float,float] = (0. , 1.) ,
              radius_for_invalid_winsor : float = 1e4):
    """Multi-step trim, clip, and robust winsorization pipeline.

    Args:
        v: 1-D numeric array.
        center: ``'median'`` or ``'mean'`` for the robust center.
        scale: ``'mad'`` or ``'sd'`` for spread around center.
        const: Multiplier for ``center ± const * scale`` clip; default from ``(center, scale)`` pair.
        trim_val: Optional ``(low, high)`` passed to ``trim``.
        winsor_val: Optional ``(low, high)`` passed to ``winsor``.
        winsor_pct: Final quantile clip ``(q_low, q_high)``.
        radius_for_invalid_winsor: Fallback scale when robust spread is zero.

    Returns:
        Winsorized 1-D array.

    Raises:
        AssertionError: Invalid ``center``/``scale``.
        KeyError: Invalid default ``const`` for mixed center/scale choice.
    """
    assert center in ['median' , 'mean'] , center
    assert scale in ['mad' , 'sd'] , scale
    
    v = trim(v , *trim_val)
    v = winsor(v , *winsor_val)

    center_val = np.nanmedian(v) if center == 'median' else  np.nanmean(v)
    radius_val = np.nanmedian(np.abs(v - np.nanmedian(v))) if scale == 'mad' else np.nanstd(v)
    if radius_val == 0:
        radius_val = radius_for_invalid_winsor

    if const is None:
        if center == 'median' and scale == 'mad' : 
            const = 5.
        elif center == 'mean' and scale == 'sd' : 
            const = 3.5
        else: 
            raise KeyError(center , scale)

    v = winsor(v , center_val - const * radius_val , center_val + const * radius_val)
    v = winsor(v , np.quantile(v , winsor_pct[0]) , np.quantile(v , winsor_pct[1]))

    return v

def time_weight(length : int , halflife : int = 0):
    """Exponential decay weights over ``length`` periods (recent = larger weight).

    Args:
        length: Number of periods.
        halflife: If positive, decay half-life in periods; if non-positive, uniform weights.

    Returns:
        1-D weights normalized to mean 1.
    """
    if halflife > 0:
        wgt = np.exp(np.log(0.5) * np.flip(np.arange(length)) / halflife)
    else:
        wgt = np.ones(length)
    wgt /= np.mean(wgt)
    return wgt

def descriptor(v : pd.Series , whiten_weight , fillna : Literal['min','max','median'] | float , group = None) -> pd.Series:
    """Winsorize, whiten, then fill remaining NaNs.

    Args:
        v: Input series.
        whiten_weight: Weights for ``whiten`` after ``winsorize``.
        fillna: ``'min'``, ``'max'``, ``'median'``, ``'mean'``, or a numeric constant.
        group: Optional DataFrame with ``indus`` column for group-wise fill.

    Returns:
        Processed ``pd.Series``.
    """
    v = whiten(winsorize(v) , whiten_weight)
    if fillna in ['max' , 'min' , 'median' , 'mean']:
        if group is not None:
            fillv = v.to_frame(name='value').join(group).groupby('indus').transform(fillna)['value']
        else:
            fillv = getattr(np , f'nan{fillna}')(v)
    else:
        fillv = fillna
    return v.where(~v.isna() , fillv)

def _lstsq_rst(x : np.ndarray , y : np.ndarray):
    """Internal: ``scipy.linalg.lstsq`` with error fallback to zeros.

    Args:
        x: Design matrix ``(n, k)``.
        y: Targets ``(n, m)`` or ``(n,)``.

    Returns:
        Least-squares coefficients or zeros ``(k, 1)`` on failure.
    """
    lstsq_result = lstsq(x, y)
    if lstsq_result is None:
        alert_message(f'lstsq error! x : {x} , y : {y}' , color = 'lightred')
        return np.zeros((x.shape[-1],1))
    return lstsq_result[0]
    
def apply_ols(x : np.ndarray | pd.DataFrame | pd.Series , y : np.ndarray | pd.DataFrame | pd.Series , 
              time_weight = None , intercept = True , respective = False):
    """Weighted OLS coefficients (multi-target).

    Args:
        x: Regressors ``(n, k)`` or coercible from pandas; 1-D becomes column vector.
        y: Targets ``(n, m)``.
        time_weight: Optional length-``n`` weights (normalized to mean 1).
        intercept: If True, prepend a column of ones.
        respective: If True, fit one regressor per column of ``y`` (requires ``x.shape[1] == m``).

    Returns:
        Coefficient matrix ``(k_or_k+1, m)`` with NaN columns where ``y`` is all-NaN.

    Raises:
        AssertionError: On shape mismatch.
    """
    if isinstance(x , (pd.Series | pd.DataFrame)): 
        x = x.to_numpy()
    if isinstance(y , (pd.Series | pd.DataFrame)): 
        y = y.to_numpy()
    if x.ndim == 1: 
        x = x[:,None]
    assert x.ndim == 2 and y.ndim == 2 , (x.shape , y.shape)
    assert len(x) == len(y) , (x.shape , y.shape)
    n_vars = y.shape[-1]
    all_nan = np.all(np.isnan(y) , axis = 0)
    if respective: 
        assert x.shape[-1] == n_vars , (x.shape , n_vars)
        all_nan *= np.all(np.isnan(x) , axis = 0)
    wgt = np.ones(len(x)) if time_weight is None else time_weight / time_weight.mean()

    y_weighted = np.nan_to_num(y * wgt[:,None] / wgt[:,None].mean(axis=0 , keepdims=True))
    x_weighted = np.nan_to_num(x) * wgt[:,None]

    if respective:
        x_weighted = x_weighted[:,None]
        if intercept: 
            x_weighted = np.pad(x_weighted , ((0,0),(1,0),(0,0)) , constant_values=1)
        coef = np.concatenate([_lstsq_rst(x_weighted[...,i], y_weighted[:,i][:,None]) for i in range(n_vars)] , axis = 1)
    else:
        if intercept: 
            x_weighted = np.pad(x_weighted , ((0,0),(1,0)) , constant_values=1)
        coef = _lstsq_rst(x_weighted, y_weighted)
    coef[:,all_nan] = np.nan
    return coef

def lm_resid(y , x : np.ndarray | pd.Series | pd.DataFrame | None , weight : Any = None , normalize = True) -> Any:
    """OLS or WLS residuals with optional winsor-whitening.

    Args:
        y: Dependent variable (array-like).
        x: Regressors 1-D or 2-D, or None for residual = ``y``.
        weight: Optional observation weights for WLS.
        normalize: If True, apply ``whiten(winsorize(resid))`` to residuals.

    Returns:
        Residuals (same shape as ``y``).

    Raises:
        ValueError: If ``x`` has ndim not in ``{1, 2}``.
    """
    if x is None:
        y_hat = 0
    else:
        finite = np.isfinite(y)
        x_finite = np.isfinite(x)
        if x.ndim == 1:
            finite = finite * x_finite
        elif x.ndim == 2:
            finite = finite * x_finite.all(axis = 1)
        else:
            raise ValueError(f'x must be 1D or 2D, but got {x.ndim}')
        _x , _y = x[finite] , y[finite]
        _w = weight if weight is None else weight[finite]
        if weight is None:
            model = sm.OLS(_y , sm.add_constant(_x)).fit()
        else:
            model = sm.WLS(_y , sm.add_constant(_x) , weights = _w).fit()
        y_hat = model.predict(sm.add_constant(x))
    resid = y - y_hat
    if normalize:
        resid = whiten(winsorize(resid))
    return resid

def shrink_cov(X : np.ndarray , min_periods : int | None = None , corr = False):
    """Ledoit–Wolf style shrinkage covariance on ``X`` (columns = variables).

    Args:
        X: ``(n, p)`` data with NaNs allowed (masked in inner products).
        min_periods: If set, zero out rows/cols with fewer than this many finite observations.
        corr: If True, convert covariance to correlation via ``cov_to_corr``.

    Returns:
        ``(p, p)`` shrunk covariance (or correlation).
    """
    n , p = X.shape
    Q = np.isfinite(X) * 1
    X = np.nan_to_num(X - np.nanmean(X , axis = 0 , keepdims=True))
    S = X.T.dot(X) / (Q.T.dot(Q) - 1)
    m  = S.diagonal().mean()
    d2 = ((S - m * np.eye(p)) ** 2).sum()
    b_bar2 = np.sum([np.square(v[:,None].dot(v[None]) - S).sum() for v in X]) / (Q.T.dot(Q) - 1).sum()
    lamb = (d2 - min(d2, b_bar2)) / d2
    cov =  (1-lamb) * m * np.eye(p) + lamb * S
    if min_periods:
        idx = Q.sum(axis = 0) >= min_periods
        cov[~idx] = np.nan
        cov[:,~idx] = np.nan
    if corr: 
        cov = cov_to_corr(cov)
    return(cov)

def normal_cov(X : np.ndarray , min_periods : int | None = None , corr = False):
    """Sample covariance via ``pandas.DataFrame.cov``.

    Args:
        X: ``(n, p)`` data.
        min_periods: ``min_periods`` for pandas covariance.
        corr: If True, convert to correlation.

    Returns:
        ``(p, p)`` covariance or correlation matrix.
    """
    cov = pd.DataFrame(X).cov(min_periods=min_periods).values
    if corr: 
        cov = cov_to_corr(cov)
    return(cov)

def cov_to_corr(cov : np.ndarray):
    """Scale covariance to correlation using diagonal standard deviations.

    Args:
        cov: Square covariance matrix.

    Returns:
        Correlation matrix of the same shape.
    """
    sd = np.sqrt(cov.diagonal())[None]
    return cov / sd.T.dot(sd)

def weighted_ts(ts : np.ndarray , nwindow : int = 504 , halflife : int | None = None):
    """Last ``nwindow`` rows of ``ts`` multiplied by exponential time weights.

    Args:
        ts: ``(T, p)`` time series panel.
        nwindow: Maximum history length to keep.
        halflife: If set, exponential half-life in row units; else uniform weights.

    Returns:
        Weighted tail slice ``(min(T, nwindow), p)``.
    """
    n = min(len(ts) , nwindow)
    if halflife:
        wgt = np.exp(np.log(0.5) * np.flip(np.arange(n)) / halflife)[:,None]
        wgt /= np.mean(wgt)
    else:
        wgt = 1
    return ts[-n:] * wgt

def ewma_cov(ts , nwindow : int = 504 , halflife : int | None = None , shrinkage : float = 0.33 , 
             corr = False):
    """Blend sample and shrunk covariance on a weighted recent window.

    Args:
        ts: ``(T, p)`` data.
        nwindow: Window length for ``weighted_ts``.
        halflife: Optional exponential half-life.
        shrinkage: Weight on ``shrink_cov`` in ``[0, 1]``.
        corr: If True, return correlation matrices.

    Returns:
        ``(p, p)`` blended covariance (or correlation).

    Raises:
        AssertionError: If ``shrinkage`` not in ``[0, 1]``.
    """
    assert 0 <= shrinkage <= 1 , shrinkage
    min_periods=int(nwindow / 4)
    ts = weighted_ts(ts , nwindow , halflife)
    v = normal_cov(ts , min_periods , corr)
    if shrinkage > 0: 
        v = v * (1 - shrinkage) + shrink_cov(ts , min_periods , corr) * shrinkage
    return v

def ewma_sd(ts , nwindow : int = 504 , halflife : int | None = None):
    """Per-column nan std on a weighted recent window with coverage mask.

    Args:
        ts: ``(T, p)`` data.
        nwindow: Window length.
        halflife: Optional half-life for ``weighted_ts``.

    Returns:
        1-D array of length ``p``; NaN where finite count per column ``< nwindow/4``.
    """
    min_periods=int(nwindow / 4)
    ts = weighted_ts(ts , nwindow , halflife)
    v = np.nanstd(ts , axis = 0)
    v[np.isfinite(ts).sum(axis = 0) < min_periods] = np.nan
    return v
