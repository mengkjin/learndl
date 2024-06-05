import numpy as np
from scipy.stats import rankdata


def ma(x):
    return np.nanmean(x, axis=0)


def ts_argmax(x, min_n):
    nan_num = np.isnan(x).sum(axis=0)
    flg = nan_num < (len(x) - min_n)
    argmax = np.nanargmax(x[:, flg], axis=0)
    rtn = np.full(x.shape[1], -1, dtype=int)
    rtn[flg] = argmax
    return rtn


def ts_argmin(x, min_n):
    nan_num = np.isnan(x).sum(axis=0)
    flg = nan_num < (len(x) - min_n)
    argmin = np.nanargmin(x[:, flg], axis=0)
    rtn = np.full(x.shape[1], -1, dtype=int)
    rtn[flg] = argmin
    return rtn


def ts_sum(x):
    return np.nansum(x, axis=0)


def lma(x):
    n = x.shape[0]
    w = np.arange(1, n + 1)
    w = w / np.sum(w)
    rtn = np.nanmean(x.T * w, axis=1) * x.shape[0]
    return rtn


def slope(x):
    nan_flg = np.isnan(x).all(axis=0)
    x_dm = x - np.nanmean(x, axis=0)
    #
    t = np.arange(x.shape[0], dtype=np.float)
    t -= np.mean(t)
    t = t / np.std(t)
    rtn = np.nanmean(t * x_dm.T, axis=1)
    rtn[nan_flg] = np.nan
    return rtn


def std(x, w=None):
    if w is None:
        rtn = np.nanstd(x, axis=0)
    else:
        dmx = (x - np.nanmean(x, axis=0)) ** 2
        rtn = np.sqrt(np.nanmean(dmx.T * w, axis=1) * x.shape[0] / np.sum(w))
    return rtn


def nrm_range(x):
    return (np.nanmax(x, axis=0) - np.nanmin(x, axis=0)) / np.nanmean(x, axis=0)


def cor_with_avg(x):
    m = np.nanmean(x, axis=1)
    dmx = x - np.nanmean(x, axis=0)
    dmm = m - np.nanmean(m)
    dmm_std = np.nanstd(dmm)
    assert dmm_std > 0.0
    crs = np.nanmean(dmx.T * dmm, axis=1)
    dmx_std = np.nanstd(dmx, axis=0)
    cor = crs / dmx_std / dmm_std
    cor[dmx_std <= 0.0] = np.nan
    rtn = cor
    return rtn


def quantile(x):
    rtn = rankdata(x, axis=0)[-1] / len(x)
    return rtn