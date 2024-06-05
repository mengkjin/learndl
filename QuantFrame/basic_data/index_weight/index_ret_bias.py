from basic_src_data.wind_tools.index import get_index_level_by_wind
from index_weight.api import load_index_weight_data
from daily_bar.api import load_daily_bar_data
from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd


def calc_wt_index_ret_bias(root_path, bm_index, scd, ecd):
    bm_scd = CALENDAR_UTIL.get_latest_n_trading_dates(scd, 2)[0]
    bm_index_ret = get_index_level_by_wind(bm_scd, ecd, bm_index).set_index(["CalcDate"])
    bm_index_ret["level_ret"] = bm_index_ret["close_level"] / bm_index_ret["close_level"].shift(1) - 1
    #
    daily_bar = load_daily_bar_data(root_path, "basic", bm_scd, ecd)[["CalcDate", "Code", "ret"]]
    index_weight_data = load_index_weight_data(
        root_path, "broad_based", [bm_index], bm_scd,
        CALENDAR_UTIL.get_last_trading_dates([ecd], inc_self_if_is_trdday=False)[0]
                                               ).rename(columns={bm_index: "weight"}, errors="raise")
    index_weight_data["CalcDate"] = CALENDAR_UTIL.get_next_trading_dates(index_weight_data["CalcDate"],
                                                                         inc_self_if_is_trdday=False)
    index_stk_ret = pd.merge(index_weight_data, daily_bar, on=["CalcDate", "Code"], how="left")
    index_stk_ret["wt_ret"] = index_stk_ret["weight"] * index_stk_ret["ret"]
    wt_stk_index_ret = index_stk_ret.groupby(["CalcDate"])["wt_ret"].sum(min_count=1)
    #
    rtn = pd.concat((bm_index_ret.reindex(wt_stk_index_ret.index), wt_stk_index_ret), axis=1, sort=True).reset_index(drop=False)
    rtn["index_ret_bias"] = rtn["wt_ret"] - rtn["level_ret"]
    rtn = rtn[rtn["CalcDate"].between(scd, ecd, inclusive="both")].set_index(["CalcDate"])["index_ret_bias"].copy()
    return rtn