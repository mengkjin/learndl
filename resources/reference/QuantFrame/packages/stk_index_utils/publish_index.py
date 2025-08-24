from daily_bar.api import load_daily_bar_data
from events_system.calendar_util import CALENDAR_UTIL
from index_weight.api import load_index_weight_data
from index_level.api import load_index_level_data
import pandas as pd


WEIGHT_EPSILON = 1e-8


INDEX_WEIGHT = dict()
weight_init_buffer = 10


def load_pub_index_weight_data(root_path, index_nm, scd, ecd):
    rtn = load_index_weight_data(root_path, "broad_based", [index_nm], scd, ecd)
    rtn.rename(columns={index_nm: 'member_weight'}, errors="raise", inplace=True)
    rtn = rtn[rtn["member_weight"] > WEIGHT_EPSILON].copy()
    assert rtn["CalcDate"].unique().tolist() == CALENDAR_UTIL.get_ranged_trading_dates(rtn["CalcDate"].min(), rtn["CalcDate"].max())
    return rtn


def load_pub_index_level(root_path, index_nm, scd, ecd):
    rtn = load_index_level_data(root_path, scd, ecd, "broad_based", [index_nm])
    rtn.drop(columns=["index_code"], inplace=True)
    return rtn


def calc_pub_index_ret_bias(root_path, index_nm, scd, ecd):
    bm_scd = CALENDAR_UTIL.get_latest_n_trading_dates(scd, 2)[0]
    bm_index_ret = load_pub_index_level(root_path, index_nm, bm_scd, ecd).set_index(["CalcDate"])
    #
    bm_index_ret["level_ret"] = bm_index_ret["close_level"] / bm_index_ret["close_level"].shift(1) - 1
    daily_bar = load_daily_bar_data(root_path, "basic", bm_scd, ecd)[["CalcDate", "Code", "ret"]]
    #
    index_weight_data = load_pub_index_weight_data(
        root_path, index_nm, bm_scd, CALENDAR_UTIL.get_last_trading_dates([ecd], inc_self_if_is_trdday=False)[0])
    index_weight_data["CalcDate"] = CALENDAR_UTIL.get_next_trading_dates(index_weight_data["CalcDate"],
                                                                         inc_self_if_is_trdday=False)
    index_stk_ret = pd.merge(index_weight_data, daily_bar, on=["CalcDate", "Code"], how="left")
    index_stk_ret["ret"] = index_stk_ret["ret"].fillna(0.0)
    index_stk_ret["wt_ret"] = index_stk_ret["member_weight"] * index_stk_ret["ret"]
    wt_stk_index_ret = index_stk_ret.groupby(["CalcDate"])["wt_ret"].sum(min_count=1)
    #
    rtn = pd.concat((bm_index_ret.reindex(wt_stk_index_ret.index), wt_stk_index_ret), axis=1, sort=True).reset_index(drop=False)
    rtn["index_ret_bias"] = rtn["wt_ret"] - rtn["level_ret"]
    #
    rtn = rtn[rtn["CalcDate"].between(scd, ecd, inclusive="both")].set_index(["CalcDate"])["index_ret_bias"].copy()
    assert rtn.notna().all()
    return rtn