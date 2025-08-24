import numpy as np
import pandas as pd
from .fcst_stats_utils.ic_perf import calc_ic
from events_system.calendar_util import CALENDAR_UTIL


def calc_ic_perf(factor_val_df, ret_type, price_type, yend_ed, bm_index_nm, ic_type):
    ret_range_type = "period"
    lag = 0
    rtn = calc_ic(factor_val_df, ret_type, yend_ed, bm_index_nm, ret_range_type, price_type, ic_type, lag)
    return rtn


def evaluate_ic(ic_df, freq_type):
    freq_dict = {"day": 245, "week": 50, "month": 12}
    ic_mean = ic_df.mean()
    ic_abs_mean = ic_df.abs().mean()
    ic_ir = (ic_df.mean() / ic_df.std()) * np.sqrt(freq_dict[freq_type])
    rtn = pd.concat((ic_mean.rename("ic_mean"), ic_abs_mean.rename("ic_abs_mean"), ic_ir.rename("ic_ir")), axis=1)
    return rtn


def evaluate_ic_yearly(ic_data, freq_type):
    factor_list = ic_data.columns.tolist()
    #
    ic_direction = np.sign(ic_data.sum())
    ic_data = ic_data * ic_direction
    #
    ic_data["year"] = ic_data.index.str[:4]
    year_rslt = ic_data.groupby(["year"])[factor_list].apply(_calc_indicator).reset_index(drop=False)
    scd, ecd = ic_data.index[0], ic_data.index[-1]
    year_rslt.loc[year_rslt["year"] == ecd[:4], "year"] = ecd.replace("-", "") + "止"
    if scd != CALENDAR_UTIL.get_ranged_trading_dates(scd[:4] + "-01-01", scd)[0]:
        year_rslt.loc[year_rslt["year"] == scd[:4], "year"] = scd.replace("-", "") + "起"
    #
    all_sample_rslt = _calc_indicator(ic_data[factor_list]).reset_index(drop=False)
    all_sample_rslt["year"] = "全样本"
    #
    freq_dict = {"day": 245, "week": 50, "month": 12}
    rtn = pd.concat((year_rslt, all_sample_rslt), axis=0)
    rtn["ic_ir"] = rtn["ic_ir"] * np.sqrt(freq_dict[freq_type])
    rtn = pd.merge(rtn, ic_direction.rename("direction"), left_on=["factor_name"], right_index=True, how="left")
    rtn = rtn[pd.Index(["factor_name", "year", "direction"]).append(rtn.columns.drop(["factor_name", "year", "direction"]))].copy()
    return rtn


def _calc_indicator(ic_se):
    ic_mean = ic_se.mean()
    ic_abs_mean = ic_se.abs().mean()
    ic_std = ic_se.std()
    ic_ir = ic_mean / ic_std
    ic_cumsum = ic_se.cumsum()
    ic_maxdown = (ic_cumsum - ic_cumsum.cummax()).min()
    rtn = pd.concat((ic_mean.rename("ic_mean"), ic_std.rename("ic_std"), ic_ir.rename("ic_ir"),
                     ic_abs_mean.rename("ic_abs_mean"), ic_maxdown.rename("ic_maxdown")), axis=1, sort=True)
    return rtn