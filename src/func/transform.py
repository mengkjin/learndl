import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

from typing import Any , Literal , Optional
from scipy.stats import norm, rankdata
from scipy.linalg import lstsq

from src.proj import Logger

warnings.filterwarnings('ignore' , category=RuntimeWarning , message='Mean of empty slice')

def fill_na_as_const(x_: np.ndarray, c_ : float = 0.0):
    rtn = x_.copy()
    rtn[np.isnan(x_)] = c_
    return rtn

def norm_to_1(x_: np.ndarray , value : float = 1.):
    if len(x_) and value:
        if np.nanmax(x_) - np.nanmin(x_) == 0:
            return np.zeros_like(x_)
        else:
            return (x_ - np.nanmin(x_)) / (np.nanmax(x_) - np.nanmin(x_)) * value
    else:
        return np.zeros_like(x_)

def rank(x_: np.ndarray, is_zscore=True, is_norm_inv=False):
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
    assert (x_ > 0).all()
    assert (x_ < 1).all()
    rtn = norm.ppf(x_)
    return rtn

def zscore(x_: np.ndarray):
    d = np.nanstd(x_)
    u = np.nanmean(x_) + 1e-6
    return (x_ - u) / d

def winsorize_by_bnd(src_: np.ndarray, u_prc_: float = 0.95, l_prc_: float = 0.05):
    assert src_.ndim == 1
    assert 0 < l_prc_ and l_prc_ < u_prc_ and u_prc_ < 100
    ulb = np.nanpercentile(src_, [u_prc_, l_prc_])
    des = src_.copy()
    des[des > ulb[0]] = ulb[0]
    des[des < ulb[1]] = ulb[1]
    return des

def winsorize_by_dist(src_: np.ndarray, m_: float = 3.5, winsor_rng : float = 0. ,dist_type_: int=0):
    # dist_type_:0, normal
    # dist_type_:1, log_normal
    assert src_.ndim == 1
    assert m_ > 0 , m_
    assert winsor_rng > 0 , winsor_rng
    assert dist_type_ in (0, 1)
    if dist_type_ == 1:
        assert np.all(np.logical_or(np.isnan(src_), src_ > 0)), "winsorize_by_dist>>src_ must be positive when dist type as 1(log_normal)"
        des = np.log(src_)
    else:
        des = src_.copy()
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
    # dist_type_:0, normal
    # dist_type_:1, log_normal
    assert bnd_ > 0 and bnd_ < 50
    assert src_.ndim == 1
    assert m_ > 0
    assert dist_type_ in (0, 1)
    if dist_type_ == 1:
        assert np.all(np.logical_or(np.isnan(src_), src_ > 0)), "winsorize_by_dist>>src_ must be positive when dist type as 1(log_normal)"
        des = np.log(src_)
    else:
        des = src_.copy()
    lu = np.nanpercentile(des, [bnd_, 100 - bnd_])
    bnded_flg = np.logical_and(des >= lu[0], des <= lu[1])
    avg = np.nanmean(des[bnded_flg])
    std = np.nanstd(des[bnded_flg])
    ub = avg + m_ * std
    lb = avg - m_ * std
    des[des > ub] = ub
    des[des < lb] = lb
    if dist_type_ == 1:
        des = np.exp(des)
    return des

def patched_by_industry(y_: np.ndarray, ind_risk_: np.ndarray, method_: int=0):
    # method 0: median, 1: mean
    assert method_ in (0, 1)
    assert y_.ndim == 1 and ind_risk_.ndim == 1
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
            print('warning>>patched_by_industry>>all nan encountered.')
            replaced_vals[np.isnan(replaced_vals)] = np.nanmedian(replaced_vals)
        assert not np.isnan(replaced_vals).any()
        y = y_.copy()
        y[nan_flag] = replaced_vals
    else:
        y = y_.copy()
    return y

def trim(v , v1 , v2):
    v = v + 0.
    if v1 is not None: 
        v[v < v1] = np.nan
    if v2 is not None: 
        v[v > v2] = np.nan
    return v

def winsor(v , v1 , v2):
    v = v + 0.
    if v1 is not None: 
        v[v < v1] = v1
    if v2 is not None: 
        v[v > v2] = v2
    return v

def weighted_mean(v , weight = None):
    if weight is not None:
        weight = np.nan_to_num(weight)
        return np.nansum(v * weight , axis = None) / (np.nansum(weight , axis = None) + 1e-6)
    else:
        return np.nanmean(v , axis = None)

def whiten(v : pd.DataFrame | pd.Series | np.ndarray , weight = None) -> Any:
    stdev = np.nanstd(v) if isinstance(v , np.ndarray) else np.nanstd(v.to_numpy().flatten()) + 1e-6
    mean = weighted_mean(v , weight)
    return (v - mean) / stdev

def winsorize(v , 
              center : Literal['median' , 'mean'] = 'median', 
              scale : Literal['mad' , 'sd'] = 'mad', 
              const : Optional[float] = None , 
              trim_val : tuple[Optional[float],Optional[float]] = (None , None) , 
              winsor_val : tuple[Optional[float],Optional[float]] = (None , None) , 
              winsor_pct : tuple[float,float] = (0. , 1.)):
    assert center in ['median' , 'mean'] , center
    assert scale in ['mad' , 'sd'] , scale
    
    v = trim(v , *trim_val)
    v = winsor(v , *winsor_val)

    s = np.nanmedian(np.abs(v - np.nanmedian(v))) if scale == 'mad' else np.nanstd(v)
    c = np.nanmedian(v) if center == 'median' else  np.nanmean(v)

    if const is None:
        if center == 'median' and scale == 'mad' : 
            const = 5.
        elif center == 'mean' and scale == 'sd' : 
            const = 3.5
        else: 
            raise KeyError(center , scale)

    v = winsor(v , c - const * s , c + const * s)
    v = winsor(v , np.quantile(v , winsor_pct[0]) , np.quantile(v , winsor_pct[1]))

    return v

def time_weight(length : int , halflife : int = 0):
    if halflife > 0:
        wgt = np.exp(np.log(0.5) * np.flip(np.arange(length)) / halflife)
    else:
        wgt = np.ones(length)
    wgt /= np.mean(wgt)
    return wgt

def descriptor(v : pd.Series , whiten_weight , fillna : Literal['min','max','median'] | float , group = None) -> pd.Series:
    v = whiten(winsorize(v) , whiten_weight)
    if fillna in ['max' , 'min' , 'median' , 'mean']:
        if group is not None:
            fillv = v.to_frame(name='value').join(group).groupby('indus').transform(fillna)['value']
        else:
            fillv = getattr(np , f'nan{fillna}')(v)
    else:
        fillv = fillna
    return v.where(~v.isna() , fillv)

def _lstsq(x : np.ndarray , y : np.ndarray):
    try:
        reg = lstsq(x , y)
        assert reg is not None , 'lstsq error'
        return reg[0]
    except Exception as e:
        Logger.error(f'lstsq error: {e}')
        return np.zeros((x.shape[-1],1))
    
def apply_ols(x : np.ndarray | pd.DataFrame | pd.Series , y : np.ndarray | pd.DataFrame | pd.Series , 
              time_weight = None , intercept = True , respective = False):
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
        coef = np.concatenate([_lstsq(x_weighted[...,i], y_weighted[:,i][:,None])[0] for i in range(n_vars)] , axis = 1)
    else:
        if intercept: 
            x_weighted = np.pad(x_weighted , ((0,0),(1,0)) , constant_values=1)
        coef = _lstsq(x_weighted, y_weighted)[0]
    coef[:,all_nan] = np.nan
    return coef

def neutral_resid(x , y , weight : Any = None , whiten = True):
    finite = np.isfinite(x) * np.isfinite(y)
    _x , _y = x[finite] , y[finite]
    _w = weight if weight is None else weight[finite]
    if weight is None:
        model = sm.OLS(_y , sm.add_constant(_x)).fit()
    else:
        model = sm.WLS(_y , sm.add_constant(_x) , weights = _w).fit()
    resid = y - model.predict(sm.add_constant(x))
    if whiten:
        resid = (resid - np.nanmean(resid)) / (np.nanstd(resid) + 1e-6)
    return resid

def shrink_cov(X : np.ndarray , min_periods : int | None = None , corr = False):
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
    cov = pd.DataFrame(X).cov(min_periods=min_periods).values
    if corr: 
        cov = cov_to_corr(cov)
    return(cov)

def cov_to_corr(cov : np.ndarray):
    sd = np.sqrt(cov.diagonal())[None]
    return cov / sd.T.dot(sd)

def weighted_ts(ts : np.ndarray , nwindow : int = 504 , halflife : Optional[int] = None):
    
    n = min(len(ts) , nwindow)
    if halflife:
        wgt = np.exp(np.log(0.5) * np.flip(np.arange(n)) / halflife)[:,None]
        wgt /= np.mean(wgt)
    else:
        wgt = 1
    return ts[-n:] * wgt

def ewma_cov(ts , nwindow : int = 504 , halflife : Optional[int] = None , shrinkage : float = 0.33 , 
             corr = False):
    assert 0 <= shrinkage <= 1 , shrinkage
    min_periods=int(nwindow / 4)
    ts = weighted_ts(ts , nwindow , halflife)
    v = normal_cov(ts , min_periods , corr)
    if shrinkage > 0: 
        v = v * (1 - shrinkage) + shrink_cov(ts , min_periods , corr) * shrinkage
    return v

def ewma_sd(ts , nwindow : int = 504 , halflife : Optional[int] = None):
    min_periods=int(nwindow / 4)
    ts = weighted_ts(ts , nwindow , halflife)
    v = np.nanstd(ts , axis = 0)
    v[np.isfinite(ts).sum(axis = 0) < min_periods] = np.nan
    return v
