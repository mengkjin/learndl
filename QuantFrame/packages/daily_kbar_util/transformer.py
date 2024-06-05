import numpy as np
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL


def _remove_by_pool(data, min_list_days):
    from basic_src_data.wind_tools.basic import get_n_listed_dates_from_winddf
    assert data["CalcDate"].is_monotonic_increasing
    stk_pool = get_n_listed_dates_from_winddf(data["CalcDate"].iloc[0], data["CalcDate"].iloc[-1])
    stk_pool = stk_pool.loc[(stk_pool["n_listed_date"] >= min_list_days) & (stk_pool["Code"].str[-2:] != "BJ"),
                            ["CalcDate", "Code"]].copy()
    rtn = pd.merge(data, stk_pool, how="inner", on=["CalcDate", "Code"], sort=True)
    return rtn


def calc_dm_rank_data(data):
    from factor_tools.cs_apply import apply_op, apply_rk
    rtn = apply_rk(data, prefix="")
    rtn = apply_op(rtn, lambda x: x - np.nanmean(x))
    rtn.rename(columns={nm: "dmrk_" + nm for nm in rtn.columns.drop(["CalcDate", "Code"])}, errors="raise", inplace=True)
    return rtn


def calc_rsknt_data(root_path, data):
    from factor_tools.cs_apply import apply_rsknt
    val_for_br_spec = data.rename(columns={"CalcDate": "CalcDate_y"}, errors="raise")
    date_df = val_for_br_spec[["CalcDate_y"]].drop_duplicates()
    date_df["CalcDate"] = CALENDAR_UTIL.get_last_trading_dates(
        date_df["CalcDate_y"].tolist(), inc_self_if_is_trdday=False, n=2)
    val_for_br_spec = pd.merge(val_for_br_spec, date_df, how="left", on=["CalcDate_y"])
    rtn = apply_rsknt(root_path, val_for_br_spec.drop(columns=["CalcDate_y"]), prefix="brs_")
    rtn = pd.merge(rtn, date_df, how="left", on=["CalcDate"]).drop(columns=["CalcDate"]).rename(
            columns={"CalcDate_y": "CalcDate"}, errors="raise")
    return rtn


def calc_transform_daily_kbar(root_path, daily_bar_data, to_apply_flds, keep_origin):
    min_list_days = 30
    pool_data = _remove_by_pool(daily_bar_data[daily_bar_data["is_traded"] == 1], min_list_days)
    #
    rsknt_data = calc_rsknt_data(root_path, pool_data[["CalcDate", "Code"] + to_apply_flds])
    rtn = pd.merge(daily_bar_data, rsknt_data, how="left", on=["CalcDate", "Code"], sort=True)
    #
    rank_data = calc_dm_rank_data(pool_data[["CalcDate", "Code"] + to_apply_flds])
    rtn = pd.merge(rtn, rank_data, how="left", on=["CalcDate", "Code"], sort=True)
    #
    if not keep_origin:
        rtn = rtn[["CalcDate", "Code"] + ["dmrk_{0}".format(nm) for nm in to_apply_flds] +
                  ["brs_{0}".format(nm) for nm in to_apply_flds]].copy()
    return rtn