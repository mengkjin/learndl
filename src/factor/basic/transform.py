
import numpy as np
from scipy.stats import norm, rankdata

def fill_na_as_const(x_: np.ndarray, c_ : float = 0.0):
    rtn = x_.copy()
    rtn[np.isnan(x_)] = c_
    return rtn

def norm_to_1(x_: np.ndarray):
    return (x_ - np.nanmin(x_)) / (np.nanmax(x_) - np.nanmin(x_))

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
    if winsor_rng > 0:
        des[des > ub] = ub + norm_to_1(des[des > ub]) * winsor_rng
        des[des < lb] = lb + (norm_to_1(des[des > ub]) - 1) * winsor_rng
    else:
        des[des > ub] = ub
        des[des < lb] = lb
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
