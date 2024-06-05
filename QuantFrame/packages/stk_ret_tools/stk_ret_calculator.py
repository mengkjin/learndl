from daily_bar.api import load_daily_bar_data
import pandas as pd
import numpy as np
from events_system.calendar_util import CALENDAR_UTIL


def calc_stk_period_ret(root_path, stk_info):
    assert stk_info.columns.equals(pd.Index(["Code", "y_start", "y_end"]))
    syd, eyd = stk_info["y_start"].min(), stk_info["y_end"].max()
    assert set(stk_info["y_start"]).issubset(CALENDAR_UTIL.get_ranged_trading_dates(syd, eyd)) and \
           set(stk_info["y_end"]).issubset(CALENDAR_UTIL.get_ranged_trading_dates(syd, eyd))
    #
    daily_bar = load_daily_bar_data(root_path, "basic", syd, eyd)
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
    rtn["prd_lg_ret"] = rtn["stk_end_cum_ret"] - rtn["stk_start_cum_ret"]
    rtn = rtn[["Code", "y_start", "y_end", "prd_lg_ret"]].copy()
    return rtn