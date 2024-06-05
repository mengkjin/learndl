import numpy as np
import bottleneck as bn


def ts_sum_abs_std_adj(x):
    """
    x_sum_abs_adj: x的总和 / x绝对值的和
    x_sum_std_adj: x的总和 / x的标准差
    """
    x_sum = bn.nansum(x, axis=0)
    abs_x_sum = bn.nansum(np.abs(x), axis=0)
    x_std = bn.nanstd(x, axis=0)
    #
    x_sum_abs_adj = x_sum / np.where(abs_x_sum != 0.0, abs_x_sum, np.nan)
    x_sum_std_adj = x_sum / np.where(x_std != 0.0, x_std, np.nan)
    rtn = np.vstack((x_sum_abs_adj, x_sum_std_adj))
    return rtn


def ts_std(x):
    """
    rtn: 标准差
    """
    rtn = bn.nanstd(x, axis=0).reshape(1, -1)
    return rtn


def ts_sum(x):
    """
    rtn: 总和
    """
    rtn = bn.nansum(x, axis=0).reshape(1, -1)
    return rtn


def ts_mean(x):
    """
    rtn: 均值
    """
    rtn = bn.nanmean(x, axis=0).reshape(1, -1)
    return rtn


def ts_max_min(x):
    """
    min_x: x的最小值
    max_x: x的最大值
    """
    min_x = np.nanmin(x, axis=0)
    max_x = np.nanmax(x, axis=0)
    rtn = np.vstack((min_x, max_x))
    return rtn


def ts_corr(x, y):
    x = np.where(np.isnan(y), np.nan, x)
    y = np.where(np.isnan(x), np.nan, y)
    x_dm = x - bn.nanmean(x, axis=0).reshape(1, -1)
    y_dm = y - bn.nanmean(y, axis=0).reshape(1, -1)
    #
    std_product = bn.nanstd(x_dm, axis=0) * bn.nanstd(y_dm, axis=0)
    rtn = bn.nanmean(x_dm * y_dm, axis=0) / np.where(std_product == 0.0, np.nan, std_product)
    rtn = rtn.reshape(1, -1)
    return rtn


def ts_tgrp_top_cnt_rto(x, grp_days, top_qtile):
    """
    top_freq: 沿时序对x分组，每组数量为grp_days,对于每一组，在截面上按x之和进行排序，计算x之和处于top_qtile的组数 / 总组数
    pos_freq: 沿时序对x分组，每组数量为grp_days,计算x之和大于0的组数 / 总组数
    """
    grp_x = x[x.shape[0] - x.shape[0] // grp_days * grp_days:, :].reshape((-1, grp_days, x.shape[1]))
    grp_sum = bn.nansum(grp_x, axis=1)
    grp_x_num = bn.nansum(~np.isnan(grp_x), axis=1)
    grp_sum[grp_x_num < 1 - 1e-6] = np.nan
    #
    grp_cnt = bn.nansum(~np.isnan(grp_sum), axis=0)
    grp_cnt = np.where(grp_cnt != 0.0, grp_cnt, np.nan)
    top_freq = (grp_sum > np.nanquantile(grp_sum, q=top_qtile, axis=1, keepdims=True)).sum(axis=0) / grp_cnt
    rtn = top_freq.reshape(1, -1)
    return rtn


def ts_tgrp_pos_cnt_rto(x, grp_days):
    """
    pos_freq: 沿时序对x分组，每组数量为grp_days,计算x之和大于0的组数 / 总组数
    """
    grp_x = x[x.shape[0] - x.shape[0] // grp_days * grp_days:, :].reshape((-1, grp_days, x.shape[1]))
    grp_sum = bn.nansum(grp_x, axis=1)
    grp_x_num = bn.nansum(~np.isnan(grp_x), axis=1)
    grp_sum[grp_x_num < 1 - 1e-6] = np.nan
    #
    grp_cnt = bn.nansum(~np.isnan(grp_sum), axis=0)
    grp_cnt = np.where(grp_cnt != 0.0, grp_cnt, np.nan)
    pos_freq = (grp_sum > 0.0).sum(axis=0) / grp_cnt
    rtn = pos_freq.reshape(1, -1)
    return rtn


def ts_change_frm_hgh_low(x):
    """
    loss_frm_hgh: 历史累和最大值-当前累和值
    gain_frm_low: 当前累和值 - 历史累和最小值
    nrm_pos: (当前累和值 - 历史累和最小值) / (历史累和最大值 - 历史累和最小值)
    """
    cum_array = np.nancumsum(x, axis=0)
    min_x = np.min(cum_array, axis=0)
    max_x = np.max(cum_array, axis=0)
    #
    loss_frm_hgh = max_x - cum_array[-1, :]
    gain_frm_low = cum_array[-1, :] - min_x
    nrm_pos = (cum_array[-1, :] - min_x) / np.where(max_x - min_x != 0.0, max_x - min_x, np.nan)
    rtn = np.vstack((loss_frm_hgh, gain_frm_low, nrm_pos))
    return rtn