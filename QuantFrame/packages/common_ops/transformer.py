from typing import List
import numpy as np
from scipy.stats import norm, rankdata
from asm_sm_tools.stats.regress import regress, lin_predict, create_regress_x, regress1
from asm_sm_tools.ordering_tools import is_member


def fill_na_as_const(x_: np.ndarray, c_=0.0):
    rtn = x_.copy()
    rtn[np.isnan(x_)] = c_
    assert x_.ndim == rtn.ndim
    return rtn


def norm_to_1(x_: np.ndarray):
    x = x_.copy()
    nan_flg = np.isnan(x)
    median_val = np.nanmedian(x_)
    if np.isnan(median_val):
        rtn = np.full_like(x, np.nan)
    else:
        x[nan_flg] = np.nanmedian(x_)
        rtn = rankdata(x, method='dense')
        rtn = rtn.astype(float) - 1.0
        rtn = (rtn / np.max(rtn) - 0.5) * 2.0
    return rtn


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


def rank_array(x_: np.ndarray, is_zscore=True, is_norm_inv=False):
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


def rank_in_category(x_: List[np.ndarray], category_: np.ndarray):
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


def rank_in_category1(x_: List[np.ndarray], category_: np.ndarray):
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
    assert x_.ndim == 2
    d = np.nanstd(x_, axis=0)
    u = np.nanmean(x_, axis=0)
    assert (d > 0.0).all()
    rtn = (x_ - u) / d
    return rtn


def winsorize_by_bnd(src_: np.ndarray, u_prc_: float, l_prc_: float):
    assert src_.ndim == 1
    assert 0 < l_prc_ and l_prc_ < u_prc_ and u_prc_ < 100
    ulb = np.nanpercentile(src_, [u_prc_, l_prc_])
    des = src_.copy()
    des[des > ulb[0]] = ulb[0]
    des[des < ulb[1]] = ulb[1]
    return des


def winsorize_by_dist(src_: np.ndarray, m_: float, dist_type_: int=0):
    # dist_type_:0, normal
    # dist_type_:1, log_normal
    assert src_.ndim == 1
    assert m_ > 0
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
    des[des > ub] = ub
    des[des < lb] = lb
    if dist_type_ == 1:
        des = np.exp(des)
    return des


def winsorize_by_bnded_dist(src_: np.ndarray, bnd_: float, m_: float, dist_type_: int=0):
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


def patched_by_risk_mdl(y_: np.ndarray, has_mkt: bool, ind_risk_: np.ndarray, style_risk_: List[np.ndarray]):
    assert has_mkt
    y = y_.copy()
    nan_flag = np.isnan(y_)
    if np.sum(nan_flag) > 0:
        non_nan_flag = np.logical_not(nan_flag)
        x, aeq, beq, ind_list = create_regress_x([x[non_nan_flag] for x in style_risk_], has_mkt, ind_risk_[non_nan_flag])
        b = regress([y_[non_nan_flag]], x, aeq, beq)[0]
        b_ind = b[-len(ind_list):]
        b_other = b[: len(style_risk_) + 1]  # type: np.ndarray
        assert set(ind_list) == set(list(ind_risk_))

        nan_ind_risk = ind_risk_[nan_flag]
        nan_ind_list = list(np.unique(nan_ind_risk))
        I, J = is_member(nan_ind_list, ind_list, is_sorted_=True)
        assert np.all(I)
        nan_ind_b = np.zeros(len(nan_ind_list))
        nan_ind_b[I] = b_ind[J[I]]

        replaced_vals = lin_predict([np.hstack((b_other, nan_ind_b))], create_regress_x([x[nan_flag] for x in style_risk_], has_mkt, ind_risk_[nan_flag])[0])[0]
        y[nan_flag] = replaced_vals.reshape(-1)
        assert not np.any(np.isnan(y))
    return y


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


def orthogonalize(y_: List[np.ndarray], numeric_x_: List[np.ndarray], intercept_: bool, dummy_x_: np.ndarray=None):
    x, aeq, beq, dummy_list = create_regress_x(numeric_x_, intercept_, dummy_x_)
    b = regress(y_, x, aeq, beq)
    yhat = lin_predict(b, x)
    res = [y_[i] - yhat[i] for i in range(len(y_))]
    return res


def orthogonalize1(y_: List[np.ndarray], x_: np.ndarray, designed_mat_: np.ndarray, bias_: float): # TODO: bias might not be a float
    b = regress1(y_, designed_mat_, bias_)
    yhat = lin_predict(b, x_)
    return [y_[i] - yhat[i] for i in range(len(y_))]


def divide_by_not_surely_pos_denom(nominator_: np.ndarray, denominator_: np.ndarray):
    rtn = np.zeros_like(nominator_) * np.nan
    nz_flag = denominator_ != 0
    rtn[nz_flag] = nominator_[nz_flag] / denominator_[nz_flag]
    return rtn


def divide_by_cautious(n_: np.ndarray, d_: np.ndarray):
    epsilon = 0.000001
    rtn = np.zeros_like(n_) * np.nan
    flg = d_ > epsilon
    rtn[flg] = n_[flg] / d_[flg]
    return rtn