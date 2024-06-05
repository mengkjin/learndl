from port_stats.ts.risk_loading import calc_port_risk_loading
from barra_model.risk_ret_est.api import load_risk_ret, load_special_ret
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL


def _calc_port_special_ret(port_stk_daily, stk_special_ret):
    port_stk_daily = port_stk_daily.copy()
    port_stk_daily["CalcDate"] = CALENDAR_UTIL.get_next_trading_dates(
        port_stk_daily["WeightDate"].tolist(), inc_self_if_is_trdday=False)
    port_stk_special = pd.merge(port_stk_daily, stk_special_ret, on=["CalcDate", "Code"], how="inner")
    port_stk_special["wt_special_ret"] = port_stk_special["special_ret"] * port_stk_special["weight"]
    rtn = port_stk_special.groupby(["CalcDate"])["wt_special_ret"].sum()
    return rtn


def _calc_sector_ret(risk_loading, risk_ret, special_ret):
    risk_ret = risk_ret.copy()
    risk_ret["CalcDate_x"] = CALENDAR_UTIL.get_last_dates(
        CALENDAR_UTIL.get_last_trading_dates(risk_ret["CalcDate"].tolist(), inc_self_if_is_trdday=False))
    #
    factor_col = risk_loading.columns.drop(["CalcDate_x"])
    risk_loading_ret = pd.merge(risk_ret, risk_loading, on=["CalcDate_x"], how="left", suffixes=("", "_loading")
                                ).set_index(["CalcDate"])
    sector_ret = risk_loading_ret[factor_col].fillna(0.0) * risk_loading_ret[factor_col + "_loading"].fillna(0.0).values
    sector_ret = pd.concat((sector_ret, special_ret), axis=1, sort=True)
    sector_ret["total"] = sector_ret.sum(axis=1)
    return sector_ret


def calc_decomposed_port_nav(root_path, daily_port_weight, barra_type, yed):
    assert daily_port_weight.columns.equals(pd.Index(["WeightDate", "Code", "weight"]))
    port_stk_daily = daily_port_weight.sort_values(["WeightDate", "Code"])
    weight_date_list = list(port_stk_daily["WeightDate"].unique())
    date_range = CALENDAR_UTIL.get_ranged_trading_dates(weight_date_list[0], weight_date_list[-1])
    assert weight_date_list == date_range
    #
    port_stk_daily["CalcDate_x"] = CALENDAR_UTIL.get_last_dates(port_stk_daily["WeightDate"].tolist())
    #
    risk_loading = calc_port_risk_loading(root_path, port_stk_daily[["CalcDate_x", "Code", "weight"]].rename(
        columns={"CalcDate_x": "CalcDate"}, errors="raise"), barra_type)
    risk_loading.rename(columns={'CalcDate': "CalcDate_x"}, errors="raise", inplace=True)
    #
    ysd = CALENDAR_UTIL.get_fwd_n_trading_dates(port_stk_daily["WeightDate"].min(), 1, self_inclusive=False)[-1]
    risk_ret = load_risk_ret(root_path, barra_type, ysd, yed)
    risk_ret.reset_index(drop=False, inplace=True)
    #
    stk_special_ret = load_special_ret(root_path, barra_type, ysd, yed)
    port_special_ret = _calc_port_special_ret(port_stk_daily, stk_special_ret)
    port_special_ret.rename("Special", inplace=True)
    #
    rtn = _calc_sector_ret(risk_loading, risk_ret, port_special_ret)
    rtn = rtn[["total"] + rtn.columns.drop(["total"]).tolist()].copy()
    return rtn
