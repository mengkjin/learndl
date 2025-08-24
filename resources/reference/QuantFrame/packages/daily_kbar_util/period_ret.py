from daily_bar.api import load_daily_bar_data
from basic_src_data.wind_tools.index import get_index_level_by_wind
import pandas as pd
import numpy as np


def calc_period_ret(root_path, timeline):
    assert isinstance(timeline, list)
    timeline = sorted(timeline)
    daily_data = load_daily_bar_data(root_path, 'basic', timeline[0], timeline[-1])[['CalcDate', 'Code', 'log_ret', 'is_traded']].copy()
    daily_data = daily_data.set_index(['CalcDate', 'Code']).unstack().cumsum()
    assert not set(timeline).difference(set(daily_data.index))
    daily_data = daily_data.reindex(timeline)
    rtn = (daily_data.shift(-1) - daily_data).stack()
    rtn = rtn[rtn['is_traded'] > 0].reset_index().drop(columns=['is_traded'])
    return rtn


def calc_index_period_ret(root_path, timeline, index_name):
    assert isinstance(timeline, list)
    timeline = sorted(timeline)
    a = get_index_level_by_wind(timeline[0], timeline[-1], index_name)
    a.set_index(['CalcDate'], inplace=True)
    rtn = a.reindex(timeline).pct_change().shift(-1).iloc[:-1]
    rtn.rename(columns={'close_level': 'index_ret'}, inplace=True)
    return rtn


def calc_stk_period_ret(root_path, stk_info):
    assert stk_info.columns.equals(pd.Index(["Code", "y_start", "y_end"]))
    assert stk_info['y_start'].is_monotonic and stk_info['y_end'].is_monotonic
    syd, eyd = stk_info["y_start"].iloc[0], stk_info["y_end"].iloc[-1]
    daily_bar = load_daily_bar_data(root_path, "basic", syd, eyd)
    date_set = set(daily_bar['CalcDate'])
    assert set(stk_info["y_start"]).issubset(date_set) and set(stk_info["y_end"]).issubset(date_set)
    daily_bar = daily_bar[daily_bar["Code"].isin(stk_info["Code"].unique())].copy()
    daily_bar["lg_ret"] = np.log(daily_bar["close_price"] / daily_bar["prev_close"])
    daily_ret = daily_bar.set_index(["CalcDate", "Code"])["lg_ret"].unstack().fillna(0.0)
    stock_cum_ret = daily_ret.cumsum(axis=0).stack().rename("stk_cum_ret").reset_index(drop=False)
    #
    rtn = pd.merge(
        stk_info,
        stock_cum_ret.rename(columns={"CalcDate": "y_start", "stk_cum_ret": "stk_start_cum_ret"}, errors="raise"),
        on=["y_start", "Code"], how="left")
    rtn = pd.merge(
        rtn,
        stock_cum_ret.rename(columns={"CalcDate": "y_end", "stk_cum_ret": "stk_end_cum_ret"}, errors="raise"),
        on=["y_end", "Code"], how="left")
    rtn["prd_log_ret"] = rtn["stk_end_cum_ret"] - rtn["stk_start_cum_ret"]
    rtn = rtn[["Code", "y_start", "y_end", "prd_log_ret"]].copy()
    return rtn