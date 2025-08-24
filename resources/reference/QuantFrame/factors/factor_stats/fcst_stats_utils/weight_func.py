import numpy as np


all_weight_types = ["long_short", "top", "top100", "long", "short"]


def get_weights(x, weight_type, direction, grp_num):
    assert weight_type in all_weight_types
    if weight_type == 'top':
        x = (x / np.nanstd(x, axis=0)) * direction
        ptile = x.quantile(0.9)
        pflg = x > ptile
        rtn = pflg / np.sum(pflg, axis=0)
    elif weight_type == 'top100':
        top_num = 100
        x = (x / np.nanstd(x, axis=0)) * direction
        ptile = x.quantile(1 - top_num / x.shape[0])
        pflg = x > ptile
        rtn = pflg / np.sum(pflg, axis=0)
    elif weight_type == 'long':
        group_num = grp_num
        x = (x / np.nanstd(x, axis=0)) * direction
        top_pflg = x > x.quantile(1 - 1 / group_num)
        rtn = top_pflg / np.sum(top_pflg, axis=0)
    elif weight_type == 'short':
        group_num = grp_num
        x = (x / np.nanstd(x, axis=0)) * direction
        bottom_pflg = x < x.quantile(1 / group_num)
        rtn = bottom_pflg / np.sum(bottom_pflg, axis=0)
    elif weight_type == 'long_short':
        group_num = grp_num
        x = (x / np.nanstd(x, axis=0)) * direction
        top_pflg = x > x.quantile(1 - 1 / group_num)
        top_weight = top_pflg / np.sum(top_pflg, axis=0)
        bottom_pflg = x < x.quantile(1 / group_num)
        bottom_weight = bottom_pflg / np.sum(bottom_pflg, axis=0)
        rtn = top_weight - bottom_weight
    else:
        assert False, "  error::factor_stats>>fcst_stats_utils>>unknown weight type:{0}.".format(weight_type)
    return rtn