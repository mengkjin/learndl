import numpy as np
import pandas as pd


EPSILON = 1e-12


def calc_half_life_weight(window, half_life):
    half_life_wt = np.power(0.5, [(window - t) / half_life for t in range(1, window + 1)])
    half_life_wt = half_life_wt / half_life_wt.sum()
    return half_life_wt


def calc_wt_corr(ret_array, half_life, min_wt_sum):
    assert isinstance(ret_array, np.ndarray)
    f_r = ret_array - np.nanmean(ret_array, axis=0, keepdims=True)
    nan_ret_flg = np.isnan(f_r)
    f_r = np.where(nan_ret_flg, 0, f_r)
    ones_array = np.where(nan_ret_flg, 0, 1)
    half_life_wt = calc_half_life_weight(ret_array.shape[0], half_life)
    wt = ones_array * half_life_wt.reshape(-1, 1)
    #
    wt_sum = wt.sum(axis=0, keepdims=True)
    wt_for_vol = wt / np.where(wt_sum < min_wt_sum, np.nan, wt_sum)
    vol = np.sqrt((wt_for_vol * f_r * f_r).sum(axis=0, keepdims=True))
    #
    cross_wt_sum = (np.sqrt(wt).T.dot(np.sqrt(wt)))
    sqrt_wt_fr = np.sqrt(wt) * f_r
    cov_matrix = sqrt_wt_fr.T.dot(sqrt_wt_fr) / np.where(cross_wt_sum < min_wt_sum, np.nan, cross_wt_sum)
    #
    vol_sq = vol.T.dot(vol)
    corr_matrix = cov_matrix / np.where(np.abs(vol_sq) < EPSILON, np.nan, vol_sq)
    return corr_matrix


def calc_wt_vol(ret_array, half_life, min_wt_sum):
    assert isinstance(ret_array, np.ndarray)
    f_r = ret_array - np.nanmean(ret_array, axis=0, keepdims=True)
    nan_ret_flg = np.isnan(f_r)
    f_r = np.where(nan_ret_flg, 0, f_r)
    ones_array = np.where(nan_ret_flg, 0, 1)
    #
    half_life_wt = calc_half_life_weight(ret_array.shape[0], half_life)
    wt = ones_array * half_life_wt.reshape(-1, 1)
    wt_sum = wt.sum(axis=0, keepdims=True)
    wt = wt / np.where(wt_sum < min_wt_sum, np.nan, wt_sum)
    #
    volatility = np.sqrt((wt * f_r * f_r).sum(axis=0, keepdims=False))
    return volatility


def estimate_cov(risk_ret):
    assert isinstance(risk_ret, pd.DataFrame)
    factor_list = risk_ret.columns.tolist()
    date_list = risk_ret.index.tolist()
    risk_ret_array = np.array(risk_ret)
    CORR_HALF_LIFE = 252
    CORR_ROLLING_WINDOW = 504
    VOL_HALF_LIFE = 126
    VOL_ROLLING_WINDOW = 252
    MIN_WT_SUM = 0.3
    rtn = list()
    T = max(CORR_ROLLING_WINDOW, VOL_ROLLING_WINDOW)
    for i in range(T - 1, risk_ret_array.shape[0]):
        ret_data = risk_ret.iloc[i - T + 1: i + 1, :]
        corr_matrix = calc_wt_corr(ret_data.iloc[-CORR_ROLLING_WINDOW:, :].to_numpy(),
                                   CORR_HALF_LIFE, MIN_WT_SUM)
        volatility = calc_wt_vol(ret_data.iloc[-VOL_ROLLING_WINDOW:, :].to_numpy()
                                 , VOL_HALF_LIFE, MIN_WT_SUM)
        F_0 = volatility.T.dot(volatility) * corr_matrix
        rtn.append(F_0)
    rtn = pd.DataFrame(np.vstack(rtn),
                       index=pd.MultiIndex.from_product((date_list[T-1:], factor_list), names=["CalcDate", "FactorName"]),
                       columns=factor_list)
    return rtn