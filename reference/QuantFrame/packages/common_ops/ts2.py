import numpy as np


def corr(x_, y_):
    ts0 = x_
    ts1 = y_
    ts0_std = np.nanstd(ts0, axis=0)
    ts1_std = np.nanstd(ts1, axis=0)
    ts0_mean = np.nanmean(ts0, axis=0)
    ts1_mean = np.nanmean(ts1, axis=0)
    cov = np.nanmean((ts0 - ts0_mean) * (ts1 - ts1_mean), axis=0)
    num_flg = np.logical_not(
        np.any(
            [np.isnan(ts0_std), np.isnan(ts1_std), ts0_std <= 0.0, ts1_std <= 0.0],
            axis=0
        )
    )
    rtn = np.full_like(ts0_mean, np.nan)
    rtn[num_flg] = cov[num_flg] / ts0_std[num_flg] / ts1_std[num_flg]
    return rtn