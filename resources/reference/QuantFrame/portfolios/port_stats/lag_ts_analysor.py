from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd
import numpy as np
from stk_index_utils.api import load_index_level
from port_backtester.deli_backtest import calc_portfolio_eod_info
from port_generals.weight_utils import calc_port_eod_weight


def _shift_calc_dates(date_info, range_type, lag_num):
    assert lag_num >= 0
    assert not date_info.duplicated(["TradeDate"]).any() and date_info["TradeDate"].is_monotonic_increasing
    if range_type == "period":
        date_info = date_info.reset_index(drop=True)
        rtn = pd.concat((date_info["CalcDate"],
                         date_info["TradeDate"].shift(-lag_num)
                         ), axis=1).dropna(subset=["TradeDate"], how="any")
    else:
        assert False
    return rtn


def prepare_target_weight_with_lag(target_weight, ret_range_type, lag):
    assert pd.Index(["TradeDate", "CalcDate", "Code", "target_weight"]).difference(target_weight.columns).empty
    assert target_weight["CalcDate"].tolist() == CALENDAR_UTIL.get_last_trading_dates(
        target_weight["TradeDate"].tolist(), inc_self_if_is_trdday=False)
    #
    target_weight = target_weight.sort_values(by=["CalcDate", "Code"])
    date_df = _shift_calc_dates(target_weight[["CalcDate", "TradeDate"]].drop_duplicates(), ret_range_type, lag)
    rtn = pd.merge(target_weight.drop(columns=["TradeDate"]), date_df, how="inner", on=["CalcDate"])
    assert rtn["TradeDate"].is_monotonic_increasing
    rtn["CalcDate"] = CALENDAR_UTIL.get_last_trading_dates(rtn["TradeDate"].tolist(), inc_self_if_is_trdday=False)
    return rtn


def _calc_excess_ret(root_path, bm_index, port_data):
    port_data["port_ret"] = port_data["pnl"] / port_data["pnl"].shift(1) - 1
    assert port_data.index.is_monotonic_increasing
    port_data = port_data.loc[port_data.index[1]:, "port_ret"].copy()
    calc_date_list = port_data.index.tolist()
    scd, ecd = calc_date_list[0], calc_date_list[-1]
    #
    index_scd = CALENDAR_UTIL.get_latest_n_trading_dates(scd, 2)[0]
    close_level = load_index_level(root_path, index_scd, ecd, bm_index).set_index(["CalcDate"])["close_level"]
    assert not close_level.empty
    index_ret = close_level / close_level.shift(1) - 1
    index_ret = index_ret[scd: ecd].copy()
    #
    port_index_df = pd.concat((port_data, index_ret.rename("index_ret")), axis=1)
    port_index_df["ex_ret"] = port_index_df["port_ret"] - port_index_df["index_ret"]
    rtn = port_index_df[["port_ret", "index_ret", "ex_ret"]].copy()
    return rtn


def _calc_port_ret_info(root_path, bm_index, eod_info_data, daily_cash):
    nav = eod_info_data.groupby(["CalcDate"]).apply(lambda x: x["eod_share"].dot(x["eod_price"])) + daily_cash["eod_cash"]
    pnl = nav / nav[nav.index[0]]
    port_ret_info = _calc_excess_ret(root_path, bm_index, pnl.to_frame("pnl"))
    assert port_ret_info.notna().all().all()
    port_ret_info = port_ret_info.reset_index(drop=False)
    return port_ret_info


def calc_port_perf_with_lag(root_path, target_weight, eod_info_data_lag0, daily_cash_lag0, bm_index, init_cash, trd_mtch_cfg, accnt_cfg,
                            trd_make_cfg, lag_num, ecd):
    assert lag_num >= 0
    port_ret_info_lag0 = _calc_port_ret_info(root_path, bm_index, eod_info_data_lag0, daily_cash_lag0)
    rtn = [port_ret_info_lag0.assign(lag_num="lag0")]
    for lag in range(1, lag_num + 1):
        lag_port = prepare_target_weight_with_lag(target_weight, "period", lag)
        port_daily_weight = calc_port_eod_weight(root_path, lag_port, ecd)
        eod_info_data, daily_cash = calc_portfolio_eod_info(root_path, init_cash, port_daily_weight,
                                                            trd_mtch_cfg, accnt_cfg, trd_make_cfg)
        port_ret_info = _calc_port_ret_info(root_path, bm_index, eod_info_data, daily_cash)
        port_ret_info["lag_num"] = "lag{0}".format(lag)
        rtn.append(port_ret_info)
    rtn = pd.concat(rtn, axis=0).reset_index(drop=True)
    return rtn