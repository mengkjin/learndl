import pandas as pd
import numpy as np
from stk_index_utils.api import load_index_level
from events_system.calendar_util import CALENDAR_UTIL


def _calc_indicator(port_index_df, year_days):
    port_ret = port_index_df["port_ret"].sum()
    bm_index_ret = port_index_df["index_ret"].sum()
    #
    ex_ret = port_index_df["port_ret"] - port_index_df["index_ret"]
    ex_ret_cumsum = ex_ret.cumsum()
    relative_max_down = (ex_ret_cumsum - ex_ret_cumsum.cummax()).min()
    ex_ret_vol = ex_ret.std() * np.sqrt(year_days)
    ex_ret_ir = ex_ret.mean() / ex_ret.std() * np.sqrt(year_days)
    #
    rtn = pd.Series([port_ret, bm_index_ret, relative_max_down, ex_ret_vol, ex_ret_ir],
                    index=["绝对收益", "基准收益", "相对最大回撤", "年化超额波动", "信息比率"])
    return rtn


def analyze_port_smp_ret(root_path, port_daily_df, bm_index):
    assert port_daily_df.columns.equals(pd.Index(["CalcDate", "smpl_pnl"]))
    port_daily_df = port_daily_df.sort_values(["CalcDate"])
    assert CALENDAR_UTIL.get_ranged_trading_dates(
        port_daily_df["CalcDate"].min(), port_daily_df["CalcDate"].max()) == port_daily_df["CalcDate"].tolist()
    assert port_daily_df["smpl_pnl"].notna().all()
    #
    YEAR_TRADE_DAYS = 243
    port_daily_df["port_ret"] = port_daily_df["smpl_pnl"] - port_daily_df["smpl_pnl"].shift(1)
    port_daily_ret = port_daily_df.loc[port_daily_df["CalcDate"] > port_daily_df["CalcDate"].min(),
                                       ["CalcDate", "port_ret"]].copy()
    calc_date_list = port_daily_ret["CalcDate"].tolist()
    scd, ecd = calc_date_list[0], calc_date_list[-1]
    #
    index_scd = CALENDAR_UTIL.get_latest_n_trading_dates(scd, 2)[0]
    bm_index_df = load_index_level(root_path, index_scd, ecd, bm_index).reset_index(drop=False)
    assert not bm_index_df.empty
    bm_index_df["index_ret"] = np.log(bm_index_df["close_level"] / bm_index_df["close_level"].shift(1))
    bm_index_df = bm_index_df[bm_index_df["CalcDate"].between(scd, ecd, inclusive="both")].copy()
    #
    port_index_df = pd.merge(port_daily_ret, bm_index_df[["CalcDate", "index_ret"]], on=["CalcDate"], how="left")
    port_index_df["年份"] = port_index_df["CalcDate"].str[:4]
    port_index_df.set_index(["CalcDate"], inplace=True)
    year_rslt = port_index_df.groupby(["年份"]).apply(_calc_indicator, YEAR_TRADE_DAYS).reset_index(drop=False)
    year_rslt.loc[year_rslt["年份"] == ecd[:4], "年份"] = ecd.replace("-", "") + "止"
    if scd != CALENDAR_UTIL.get_ranged_trading_dates(scd[:4] + "-01-01", scd)[0]:
        year_rslt.loc[year_rslt["年份"] == scd[:4], "年份"] = scd.replace("-", "") + "起"
    #
    all_sample_rslt = _calc_indicator(port_index_df, YEAR_TRADE_DAYS).to_frame().T
    all_sample_rslt["年份"] = "全样本"
    all_sample_rslt[["绝对收益", "基准收益"]] = all_sample_rslt[["绝对收益", "基准收益"]] * YEAR_TRADE_DAYS / len(calc_date_list)
    #
    latest_3y_sd = CALENDAR_UTIL.get_n_years_before([ecd], 3)[0]
    latest_3y_rslt = _calc_indicator(port_index_df.loc[latest_3y_sd:], YEAR_TRADE_DAYS).to_frame().T
    latest_3y_rslt["年份"] = "最近3年"
    latest_3y_rslt[["绝对收益", "基准收益"]] = latest_3y_rslt[["绝对收益", "基准收益"]] * YEAR_TRADE_DAYS / port_index_df.loc[latest_3y_sd:].shape[0]
    #
    last_5y_sd = CALENDAR_UTIL.get_n_years_before([ecd], 5)[0]
    latest_5y_rslt = _calc_indicator(port_index_df.loc[last_5y_sd:], YEAR_TRADE_DAYS).to_frame().T
    latest_5y_rslt["年份"] = "最近5年"
    latest_5y_rslt[["绝对收益", "基准收益"]] = latest_5y_rslt[["绝对收益", "基准收益"]] * YEAR_TRADE_DAYS / port_index_df.loc[last_5y_sd:].shape[0]
    #
    rtn = pd.concat((year_rslt, latest_3y_rslt, latest_5y_rslt, all_sample_rslt), axis=0)
    rtn["超额收益"] = rtn["绝对收益"] - rtn["基准收益"]
    rtn["收益回撤比"] = rtn["超额收益"] / np.abs(rtn["相对最大回撤"])
    rtn = rtn[['年份', '绝对收益', '基准收益', "超额收益", '相对最大回撤', '收益回撤比', "年化超额波动", "信息比率"]].copy()
    return rtn