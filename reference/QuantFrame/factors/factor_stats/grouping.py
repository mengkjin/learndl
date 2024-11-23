import pandas as pd
import numpy as np
from .fcst_stats_utils.grouping import calc_group_perf
from events_system.calendar_util import CALENDAR_UTIL


def _calc_indicator(ret_df):
    ret_sum = ret_df.sum()
    ret_std = ret_df.std()
    ret_ir = ret_df.mean() / ret_std
    ret_cumsum = ret_df.cumsum()
    ret_maxdown = (ret_cumsum - ret_cumsum.cummax()).min()
    rtn = pd.concat((ret_sum.rename("year_ret"), ret_std.rename("ret_std"), ret_ir.rename("ir"),
                     ret_maxdown.rename("ret_max_down")), axis=1, sort=True)
    return rtn


def evaluate_top_perf_yearly(group_perf, freq_type):
    top_group = group_perf.groupby(["group", "factor_name"])["group_ret"].sum().loc[
        [group_perf["group"].min(), group_perf["group"].max()]].reset_index(drop=False).sort_values(
        ["factor_name", "group_ret"]).drop_duplicates(["factor_name"], keep="last")
    top_perf = pd.merge(top_group[["factor_name", "group"]], group_perf, how="left", on=["factor_name", "group"])
    top_perf = top_perf.set_index(["CalcDate", "factor_name"])["group_ret"].unstack()
    #
    factor_list = top_perf.columns.tolist()
    top_perf["year"] = top_perf.index.str[:4]
    year_rslt = top_perf.groupby(["year"])[factor_list].apply(_calc_indicator).reset_index(drop=False)
    scd, ecd = top_perf.index[0], top_perf.index[-1]
    year_rslt.loc[year_rslt["year"] == ecd[:4], "year"] = ecd.replace("-", "") + "止"
    if scd != CALENDAR_UTIL.get_ranged_trading_dates(scd[:4] + "-01-01", scd)[0]:
        year_rslt.loc[year_rslt["year"] == scd[:4], "year"] = scd.replace("-", "") + "起"
    #
    all_sample_rslt = _calc_indicator(top_perf[factor_list]).reset_index(drop=False)
    all_sample_rslt["year"] = "全样本"
    freq_days_dict = {"day": 245, "week": 50, "month": 12}
    assert freq_type in freq_days_dict.keys()
    all_sample_rslt["year_ret"] = all_sample_rslt["year_ret"] / top_perf.shape[0] * freq_days_dict[freq_type]
    all_sample_rslt["ret_std"] = all_sample_rslt["ret_std"].mean() * np.sqrt(freq_days_dict[freq_type])
    #
    rtn = pd.concat((year_rslt, all_sample_rslt), axis=0)
    rtn["ir"] = rtn["ir"] * np.sqrt(freq_days_dict[freq_type])
    rtn = pd.merge(rtn, top_group[["factor_name", "group"]], on=["factor_name"], how="left")
    rtn = rtn[pd.Index(["factor_name", "year", "group"]).append(rtn.columns.drop(["factor_name", "year", "group"]))].copy()
    return rtn


def calc_grp_ret(factor_val_df, ret_type, price_type, yend_ed, grp_bm_index, group_nums):
    ret_range_type = "period"
    lag = 0
    grp_perf = calc_group_perf(factor_val_df, ret_type, yend_ed, grp_bm_index, group_nums, ret_range_type, price_type, lag)
    return grp_perf