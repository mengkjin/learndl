import pandas as pd
from stk_index_utils.api import load_index_level
import numpy as np
from events_system.calendar_util import CALENDAR_UTIL


def _calc_rel_ret_for_expand(port, index):
    index = index[port.index]
    diff = (port[-1] / port) - (index[-1] / index)
    rtn = diff.min()
    return rtn


def _calc_relative_mdd(port_nav, index_nav):
    relative_down = port_nav.expanding().apply(lambda x: _calc_rel_ret_for_expand(x, index=index_nav))
    rtn = - relative_down.min()
    return rtn


def _calc_max_draw_down(port_ret):
    return_list = port_ret.tolist()
    date_list = port_ret.index.tolist()
    end_loc = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))
    if end_loc == 0:
        start_loc = 0
    else:
        start_loc = np.argmax(return_list[:end_loc])
    #
    max_down = (return_list[start_loc] - return_list[end_loc]) / (return_list[start_loc])
    trade_interval = end_loc - start_loc + 1
    rtn = pd.Series([max_down, trade_interval, date_list[start_loc], date_list[end_loc]],
                    index=["最大回撤", "最大回撤天数", "最大回撤起始日期", "最大回撤结束日期"])
    return rtn


def _calc_indicator(port_index_df, year_days):
    port_ret = (port_index_df["port_ret"] + 1).prod() - 1
    bm_index_ret = (port_index_df["index_ret"] + 1).prod() - 1
    #
    port_nav = (port_index_df["port_ret"] + 1).cumprod()
    max_down = _calc_max_draw_down(port_nav)
    index_nav = (port_index_df["index_ret"] + 1).cumprod()
    relative_mdd = _calc_relative_mdd(port_nav, index_nav)
    #
    ex_ret = port_index_df["port_ret"] - port_index_df["index_ret"]
    lg_ex_ret_sum = (np.log(port_index_df["port_ret"] + 1) - np.log(port_index_df["index_ret"] + 1)).sum()
    ex_ret_vol = ex_ret.std() * np.sqrt(year_days)
    ex_ret_ir = ex_ret.mean() / ex_ret.std() * np.sqrt(year_days)
    #
    turnover = port_index_df["turnover"].sum()
    rtn = pd.Series([port_ret, bm_index_ret, lg_ex_ret_sum, max_down["最大回撤"], relative_mdd, max_down["最大回撤天数"], max_down["最大回撤起始日期"],
                     max_down["最大回撤结束日期"], ex_ret_vol, ex_ret_ir, turnover],
                    index=["绝对收益", "基准收益", "超额收益(对数)", "最大回撤", "相对最大回撤", "最大回撤天数", "最大回撤起始日期", "最大回撤结束日期",
                           "跟踪误差", "信息比率", "换手率"])
    return rtn


def _calc_port_perf(root_path, port_data, bm_index):
    YEAR_TRADE_DAYS = 245
    port_data["port_ret"] = port_data["pnl"] / port_data["pnl"].shift(1) - 1
    port_data = port_data.iloc[2:][["CalcDate", "port_ret", "turnover"]].copy()
    calc_date_list = port_data["CalcDate"].tolist()
    scd, ecd = calc_date_list[0], calc_date_list[-1]
    #
    index_scd = CALENDAR_UTIL.get_latest_n_trading_dates(scd, 2)[0]
    bm_index_df = load_index_level(root_path, index_scd, ecd, bm_index).reset_index(drop=False)
    assert not bm_index_df.empty
    bm_index_df["index_ret"] = bm_index_df["close_level"] / bm_index_df["close_level"].shift(1) - 1
    bm_index_df = bm_index_df[bm_index_df["CalcDate"].between(scd, ecd, inclusive="both")].copy()
    #
    port_index_df = pd.merge(port_data, bm_index_df[["CalcDate", "index_ret"]], on=["CalcDate"], how="left")
    calc_date_list = port_index_df["CalcDate"].tolist()
    port_index_df["年份"] = port_index_df["CalcDate"].str[:4]
    port_index_df.set_index(["CalcDate"], inplace=True)
    year_rslt = port_index_df.groupby(["年份"]).apply(_calc_indicator, YEAR_TRADE_DAYS).reset_index(drop=False)
    #
    scd, ecd = calc_date_list[0], calc_date_list[-1]
    year_rslt.loc[year_rslt["年份"] == ecd[:4], "年份"] = ecd.replace("-", "") + "止"
    if scd != CALENDAR_UTIL.get_ranged_trading_dates(scd[:4] + "-01-01", scd)[0]:
        year_rslt.loc[year_rslt["年份"] == scd[:4], "年份"] = scd.replace("-", "") + "起"
    #
    all_sample_rslt = _calc_indicator(port_index_df, YEAR_TRADE_DAYS).to_frame().T
    all_sample_rslt["年份"] = "全样本"
    all_sample_rslt[["绝对收益", "基准收益"]] = np.power(all_sample_rslt[["绝对收益", "基准收益"]] + 1,
                                                 YEAR_TRADE_DAYS / len(calc_date_list)) - 1
    all_sample_rslt["换手率"] = all_sample_rslt["换手率"] / len(calc_date_list) * YEAR_TRADE_DAYS
    #
    perf_rslt = pd.concat((year_rslt, all_sample_rslt), axis=0)
    perf_rslt["超额收益"] = perf_rslt["绝对收益"] - perf_rslt["基准收益"]
    perf_rslt["relative_mdd_ex_zero"] = perf_rslt["相对最大回撤"].replace(0, 1e-4) # float division by zero
    perf_rslt["相对收益回撤比"] = perf_rslt["超额收益"] / perf_rslt["relative_mdd_ex_zero"]
    perf_rslt = perf_rslt[['年份', "绝对收益", "基准收益", "超额收益", "相对最大回撤",
                           "最大回撤", "最大回撤天数", "最大回撤起始日期", "最大回撤结束日期",
                           "跟踪误差", "信息比率", "相对收益回撤比", "换手率"]].copy()
    port_index_df["ex_ret"] = port_index_df["port_ret"] - port_index_df["index_ret"]
    return perf_rslt, port_index_df


def evaluate_port_perf(root_path, eod_info_data, daily_cash, bm_index):
    assert CALENDAR_UTIL.get_ranged_trading_dates(
        eod_info_data["CalcDate"].iloc[0], eod_info_data["CalcDate"].iloc[-1]) == eod_info_data["CalcDate"].unique().tolist()
    assert CALENDAR_UTIL.get_ranged_trading_dates(daily_cash.index[0], daily_cash.index[-1]) == daily_cash.index.tolist()
    nav = eod_info_data.groupby(["CalcDate"]).apply(lambda x: x["eod_share"].dot(x["eod_price"])) + daily_cash["eod_cash"]
    pnl = nav / nav[nav.index[0]]
    #
    eod_info_data["traded_amount"] = np.abs(eod_info_data["traded_share"] * eod_info_data["traded_price"])
    eod_info_data["traded_amount"].fillna(0.0, inplace=True)
    turnover = eod_info_data.groupby(["CalcDate"])["traded_amount"].sum() / nav
    assert np.isclose(turnover[turnover.index[0]], 0.0) and turnover[turnover.index[1]] > 1e-2
    turnover[turnover.index[1]] = 0.0
    #
    port_data = pd.concat((pnl.rename("pnl"), turnover.rename("turnover")), axis=1).reset_index(drop=False)
    perf_rslt, port_index_df = _calc_port_perf(root_path, port_data, bm_index)
    return perf_rslt, port_index_df

