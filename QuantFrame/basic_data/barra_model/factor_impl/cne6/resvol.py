from daily_bar.api import load_daily_bar_data
import pandas as pd
import numpy as np
from events_system.calendar_util import CALENDAR_UTIL
from .cne6_utils import apply_zscore, apply_qtile_shrink
from daily_kbar_util.to_clip_out_limit_ret import clip_stk_ret


EPSILON = 1e-8


def _calc_hsigma_for_impl(stk_ret, mkt_ret, rolling_winsize, half_life):
    stk_ret_array = stk_ret.fillna(0.0).to_numpy()
    ones_array = stk_ret.mask(stk_ret.notna(), other=1.0).fillna(0.0).to_numpy()
    #
    mkt_ret_array = mkt_ret.to_numpy().reshape(-1, 1)
    half_life_wt = np.power(0.5, [(rolling_winsize - t) / half_life for t in range(1, rolling_winsize + 1)]).reshape(-1, 1)
    #
    date_list = stk_ret.index.tolist()
    rtn = list()
    for i in range(rolling_winsize - 1, len(date_list)):
        x = mkt_ret_array[i - rolling_winsize + 1: i + 1, :]
        y = stk_ret_array[i - rolling_winsize + 1: i + 1, :]
        wt = ones_array[i - rolling_winsize + 1: i + 1, :] * half_life_wt
        wt_sum = wt.sum(axis=0, keepdims=True)
        wt = wt / np.where(wt_sum < EPSILON, np.nan, wt_sum)
        #
        wt_x_sum = (wt * x).sum(axis=0)
        wt_x_var = (wt * x * x).sum(axis=0) - wt_x_sum ** 2
        wt_x_y_cov = (wt * x * y).sum(axis=0) - wt_x_sum * (wt * y).sum(axis=0)
        beta = wt_x_y_cov / np.where(wt_x_var < EPSILON, np.nan, wt_x_var)
        res = (y - y.mean(axis=0, keepdims=True)) - beta * (x - x.mean(axis=0, keepdims=True))
        res_vol = res.std(axis=0, keepdims=True)
        rtn.append(res_vol)
    rtn = pd.DataFrame(np.vstack(rtn),
                       index=pd.Index(date_list[rolling_winsize - 1:], name="CalcDate"),
                       columns=stk_ret.columns)
    return rtn


def calc_hsigma(root_path, scd, ecd):
    HBETA_ROLLING_WINSIZE = 490
    HBETA_HALF_LIFE = 245
    MIN_PRD_RATE = 0.5
    data_sd = CALENDAR_UTIL.get_last_trading_dates([scd], n=HBETA_ROLLING_WINSIZE,
                                                   inc_self_if_is_trdday=False)[0]
    daily_bar = load_daily_bar_data(root_path, "basic", data_sd, ecd)
    daily_bar = daily_bar.loc[(daily_bar["is_traded"] == 1) & (daily_bar["Code"].str[6:] != '.BJ'),
                              ["CalcDate", "Code", "ret"]].copy()
    daily_bar = clip_stk_ret(daily_bar)
    daily_ret = daily_bar.set_index(["CalcDate", "Code"])["ret"].unstack()
    #
    fv_data = load_daily_bar_data(root_path, "valuation",
                                  CALENDAR_UTIL.get_last_trading_dates([data_sd], inc_self_if_is_trdday=False)[0], ecd)
    fv_data = fv_data[fv_data['Code'].str[-2:] != 'BJ'].copy()
    date_df = daily_bar[["CalcDate"]].drop_duplicates()
    date_df["CalcDate_last"] = CALENDAR_UTIL.get_last_trading_dates(date_df["CalcDate"].tolist(), inc_self_if_is_trdday=False)
    daily_bar = pd.merge(daily_bar, date_df, how="left", on=["CalcDate"])
    daily_bar_fv = pd.merge(
        daily_bar[["CalcDate", "CalcDate_last", "Code", "ret"]],
        fv_data[["CalcDate", "Code", "float_value"]].rename(columns={"CalcDate": "CalcDate_last"}, errors="raise"),
        how="inner", on=["CalcDate_last", "Code"], sort=True)
    daily_bar_fv["weight"] = np.sqrt(daily_bar_fv["float_value"])
    mkt_ret = daily_bar_fv.dropna(subset=["weight", "ret"], how="any").\
        groupby(["CalcDate"]).apply(lambda x: x["ret"].dot(x["weight"]) / x["weight"].sum())
    #
    hsigma = _calc_hsigma_for_impl(daily_ret, mkt_ret, HBETA_ROLLING_WINSIZE, HBETA_HALF_LIFE)
    ret_cnt = daily_ret.rolling(HBETA_ROLLING_WINSIZE, min_periods=1).count()
    hsigma = hsigma.mask(ret_cnt.loc[hsigma.index] < MIN_PRD_RATE * HBETA_ROLLING_WINSIZE, other=np.nan)
    hsigma = hsigma.loc[scd: ecd].stack().rename("hsigma")
    return hsigma


def calc_dastd(root_path, scd, ecd):
    DASTD_ROLLING_WINSIZE = 245
    MIN_PRD_RATE = 0.5
    data_sd = CALENDAR_UTIL.get_last_trading_dates([scd], n=DASTD_ROLLING_WINSIZE,
                                                   inc_self_if_is_trdday=False)[0]
    daily_bar = load_daily_bar_data(root_path, "basic", data_sd, ecd)
    daily_bar = daily_bar.loc[(daily_bar["is_traded"] == 1) & (daily_bar["Code"].str[6:] != '.BJ'),
                              ["CalcDate", "Code", "ret"]].copy()
    daily_bar = clip_stk_ret(daily_bar)
    daily_ret = daily_bar.set_index(["CalcDate", "Code"])["ret"].unstack()
    dastd = daily_ret.rolling(DASTD_ROLLING_WINSIZE, min_periods=int(MIN_PRD_RATE * DASTD_ROLLING_WINSIZE)).std()  # TODO: may add weight
    dastd = dastd.loc[scd: ecd].stack().rename("dastd")
    return dastd


def calc_cmra(root_path, scd, ecd):
    ROLLING_WINSIZE = 245
    MIN_PRD_RATE = 0.5
    data_sd = CALENDAR_UTIL.get_last_trading_dates([scd], n=ROLLING_WINSIZE,
                                                   inc_self_if_is_trdday=False)[0]
    daily_bar = load_daily_bar_data(root_path, "basic", data_sd, ecd)
    daily_bar = daily_bar.loc[(daily_bar["is_traded"] == 1) & (daily_bar["Code"].str[6:] != '.BJ'),
                              ["CalcDate", "Code", "ret"]].copy()
    daily_ret = daily_bar.set_index(["CalcDate", "Code"])["ret"].unstack()
    #
    fv_data = load_daily_bar_data(root_path, "valuation",
                                  CALENDAR_UTIL.get_last_trading_dates([data_sd], inc_self_if_is_trdday=False)[0], ecd)
    fv_data = fv_data[fv_data['Code'].str[-2:] != 'BJ'].copy()
    date_df = daily_bar[["CalcDate"]].drop_duplicates()
    date_df["CalcDate_last"] = CALENDAR_UTIL.get_last_trading_dates(date_df["CalcDate"].tolist(), inc_self_if_is_trdday=False)
    daily_bar = pd.merge(daily_bar, date_df, how="left", on=["CalcDate"])
    daily_bar_fv = pd.merge(
        daily_bar[["CalcDate", "CalcDate_last", "Code", "ret"]],
        fv_data[["CalcDate", "Code", "float_value"]].rename(columns={"CalcDate": "CalcDate_last"}, errors="raise"),
        how="inner", on=["CalcDate_last", "Code"], sort=True)
    daily_bar_fv["weight"] = np.sqrt(daily_bar_fv["float_value"])
    mkt_ret = daily_bar_fv.dropna(subset=["weight", "ret"], how="any").\
        groupby(["CalcDate"]).apply(lambda x: x["ret"].dot(x["weight"]) / x["weight"].sum())
    #
    mkt_cum_lg_ret = np.log(mkt_ret + 1).cumsum()
    stk_cum_lg_ret = np.log(daily_ret + 1).cumsum()
    cum_lg_ex_ret = stk_cum_lg_ret.sub(mkt_cum_lg_ret, axis=0)
    cmra = cum_lg_ex_ret.rolling(ROLLING_WINSIZE, min_periods=int(MIN_PRD_RATE * ROLLING_WINSIZE)).max() - \
            cum_lg_ex_ret.rolling(ROLLING_WINSIZE, min_periods=int(MIN_PRD_RATE * ROLLING_WINSIZE)).min()
    cmra = cmra.loc[scd: ecd].stack().rename("cmra")
    return cmra


def calc_resvol(root_path, scd, ecd):
    dastd = calc_dastd(root_path, scd, ecd)
    cmra = calc_cmra(root_path, scd, ecd)
    hsigma = calc_hsigma(root_path, scd, ecd)
    #
    rtn = pd.concat((dastd, cmra, hsigma), axis=1, sort=True).reset_index()
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both")].set_index(['CalcDate', 'Code'])
    rtn = apply_qtile_shrink(np.log(rtn).dropna(how='any'))
    rtn = apply_zscore(rtn)
    rtn = apply_zscore(rtn.mean(axis=1).rename('Resvol'))
    return rtn