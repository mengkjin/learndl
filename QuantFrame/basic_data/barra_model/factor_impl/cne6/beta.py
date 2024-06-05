from daily_bar.api import load_daily_bar_data
import pandas as pd
import numpy as np
from events_system.calendar_util import CALENDAR_UTIL
from daily_kbar_util.to_clip_out_limit_ret import clip_stk_ret
from .cne6_utils import apply_qtile_shrink, apply_zscore


EPSILON = 1e-8


def _calc_beta_for_impl(stk_ret, mkt_ret, rolling_winsize, half_life):
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
        rtn.append(beta)
    rtn = pd.DataFrame(np.vstack(rtn),
                       index=pd.Index(date_list[rolling_winsize - 1:], name="CalcDate"),
                       columns=stk_ret.columns)
    return rtn


def calc_beta(root_path, scd, ecd):
    ROLLING_WINSIZE = 490
    HALF_LIFE = 245
    MIN_RATE = 0.5
    #
    data_sd = CALENDAR_UTIL.get_last_trading_dates([scd], inc_self_if_is_trdday=False, n=ROLLING_WINSIZE + 10)[0]
    daily_bar = load_daily_bar_data(root_path, "basic", data_sd, ecd)
    daily_ret = daily_bar.loc[(daily_bar["is_traded"] == 1) & (daily_bar["Code"].str[6:] != '.BJ'),
                              ["CalcDate", "Code", "ret"]].copy()
    daily_ret = clip_stk_ret(daily_ret)
    #
    fv_data = load_daily_bar_data(root_path, "valuation",
                                  CALENDAR_UTIL.get_last_trading_dates([data_sd], inc_self_if_is_trdday=False)[0], ecd)
    date_df = daily_bar[["CalcDate"]].drop_duplicates()
    date_df["CalcDate_last"] = CALENDAR_UTIL.get_last_trading_dates(date_df["CalcDate"].tolist(), inc_self_if_is_trdday=False)
    daily_bar = pd.merge(daily_bar, date_df, how="left", on=["CalcDate"])
    daily_ret_fv = pd.merge(
        daily_bar[["CalcDate", "CalcDate_last", "Code", "ret"]],
        fv_data[["CalcDate", "Code", "float_value"]].rename(columns={"CalcDate": "CalcDate_last"}, errors="raise"),
        how="inner", on=["CalcDate_last", "Code"], sort=True)
    daily_ret_fv["weight"] = np.sqrt(daily_ret_fv["float_value"])
    mkt_ret = daily_ret_fv.dropna(subset=["weight", "ret"], how="any").\
        groupby(["CalcDate"]).apply(lambda x: x["ret"].dot(x["weight"]) / x["weight"].sum())
    #
    stk_ret = daily_ret.set_index(["CalcDate", "Code"])["ret"].unstack()
    #
    beta = _calc_beta_for_impl(stk_ret, mkt_ret, ROLLING_WINSIZE, HALF_LIFE)
    ret_cnt = stk_ret.rolling(ROLLING_WINSIZE, min_periods=1).count()
    beta = beta.mask(ret_cnt.loc[beta.index] < MIN_RATE * ROLLING_WINSIZE, other=np.nan)
    #
    rtn = beta.stack().rename("Beta").reset_index(drop=False)
    #
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both")].set_index(['CalcDate', 'Code'])
    rtn = apply_qtile_shrink(rtn)
    rtn = apply_zscore(rtn['Beta'])
    return rtn
