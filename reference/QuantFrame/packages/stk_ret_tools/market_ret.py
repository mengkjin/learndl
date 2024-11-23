from daily_kbar_util.to_clip_out_limit_ret import clip_stk_ret
from daily_bar.api import load_daily_bar_data
import numpy as np
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL


def calc_sqrt_fv_wt_market_ret(root_path, scd, ecd):
    stk_ret = load_daily_bar_data(root_path, "basic", scd, ecd)
    stk_ret = stk_ret.loc[(stk_ret["is_traded"] == 1) & (stk_ret["Code"].str[6:] != '.BJ'),
                          ["CalcDate", "Code", "ret"]].copy()
    fv_data = load_daily_bar_data(root_path, "valuation",
                                  CALENDAR_UTIL.get_last_trading_dates([scd], inc_self_if_is_trdday=False)[0], ecd)
    fv_data = fv_data[fv_data['Code'].str[-2:] != 'BJ'].copy()
    stk_ret["CalcDate_last"] = CALENDAR_UTIL.get_last_trading_dates(
        stk_ret["CalcDate"].tolist(), inc_self_if_is_trdday=False)
    stk_ret = pd.merge(
        stk_ret[["CalcDate", "CalcDate_last", "Code", "ret"]],
        fv_data[["CalcDate", "Code", "float_value"]].rename(columns={"CalcDate": "CalcDate_last"}, errors="raise"),
        how="inner", on=["CalcDate_last", "Code"], sort=True)
    stk_ret["sqrt_float_value"] = np.sqrt(stk_ret["float_value"])
    stk_ret["weight"] = stk_ret["sqrt_float_value"] / stk_ret.groupby(["CalcDate"])["sqrt_float_value"].transform("sum")
    stk_ret["wt_ret"] = stk_ret["weight"] * stk_ret["ret"]
    rtn = stk_ret.groupby(["CalcDate"])["wt_ret"].sum()
    return rtn