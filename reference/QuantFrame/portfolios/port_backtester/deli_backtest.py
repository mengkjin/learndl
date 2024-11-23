from daily_bar.api import load_daily_bar_data
from divnsplit.api import load_divnsplit_data
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from daily_kbar_util.limits_price import calc_is_limited
from port_generals.to_patch_delist_earlier import get_early_delist


EPSILON = 1e-10
SHARE_EPSILON = 1e-2


def _prepare_all_data(root_path, port_daily_weight):
    assert port_daily_weight["WeightDate"].is_monotonic_increasing
    scd = port_daily_weight["WeightDate"].iloc[0]
    ecd = port_daily_weight["WeightDate"].iloc[-1]
    daily_bar = load_daily_bar_data(root_path, "basic", scd, ecd)
    dbar_close = daily_bar.set_index(["CalcDate", "Code"])[
        ["close_price"]].unstack().fillna(method="ffill").stack().reset_index(drop=False)
    daily_bar = pd.merge(daily_bar.drop(columns=["close_price"]), dbar_close, how="left", on=["CalcDate", "Code"])
    daily_bar["open_price"].fillna(daily_bar["close_price"], inplace=True)
    daily_bar["high_price"].fillna(daily_bar["close_price"], inplace=True)
    daily_bar["low_price"].fillna(daily_bar["close_price"], inplace=True)
    daily_bar["vwap"].fillna(daily_bar["close_price"], inplace=True)
    daily_bar["prev_close"].fillna(daily_bar["close_price"], inplace=True)
    daily_bar[["volume", "is_traded"]] = daily_bar[["volume", "is_traded"]].fillna(0.0)
    #
    cash_df = pd.DataFrame(
        [CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)], index=["CalcDate"]).T.assign(
        Code="CNY", is_traded=1, volume=1e8, close_price=1.0, open_price=1.0, high_price=1.0, low_price=1.0, vwap=1.0)
    daily_bar = pd.concat((daily_bar, cash_df), axis=0).sort_values(["CalcDate", "Code"]).reset_index(drop=True)
    #
    rtn = pd.merge(
        port_daily_weight[["WeightDate", "Code", "expected_weight"]].rename(columns={"WeightDate": "CalcDate"}, errors="raise"),
        daily_bar[["CalcDate", "Code", "is_traded", "open_price", "high_price", "low_price", "close_price", "vwap", "volume"]].copy(),
        how="right",
        on=["CalcDate", "Code"])
    rtn["expected_weight"].fillna(0.0, inplace=True)
    #
    limited_info = calc_is_limited(daily_bar, fld_list=("is_limit_eod", "hit_up_limit", "hit_low_limit"))
    rtn = pd.merge(rtn, limited_info, how="left", on=["CalcDate", "Code"])
    #
    divnsplit = load_divnsplit_data(root_path, scd, ecd)
    rtn = pd.merge(rtn, divnsplit[["ExDate", "Code", "split_ratio", 'div_rate']].rename(columns={"ExDate": "CalcDate"}),
                   how="left", on=["CalcDate", "Code"])
    rtn["split_ratio"].fillna(0.0, inplace=True)
    rtn["div_rate"].fillna(0.0, inplace=True)
    #
    early_delist_info = get_early_delist(root_path, scd, ecd)
    rtn = pd.merge(rtn, early_delist_info[["Date", "Code", "remove_by_delist"]].rename(columns={"Date": "CalcDate"}),
                   how="left", on=["CalcDate", "Code"])
    rtn["remove_by_delist"].fillna(0.0, inplace=True)
    #
    rtn.set_index(["CalcDate"], inplace=True)
    return rtn


def _calc_bod_port(eod_share, eod_cash, trade_list, divnsplit):
    bod_share = eod_share * (divnsplit.loc[eod_share.index, "split_ratio"] + 1)
    bod_share.rename("bod_share", inplace=True)
    trade_list["trade_share"] = trade_list["trade_share"] * (divnsplit.loc[trade_list.index, "split_ratio"] + 1)
    trade_list["expected_share"] = trade_list["expected_share"] * (divnsplit.loc[trade_list.index, "split_ratio"] + 1)
    bod_cash = eod_cash + (divnsplit.loc[eod_share.index, "div_rate"] * eod_share).sum()
    return bod_share, bod_cash, trade_list


def _make_match(bod_share, bod_cash, trade_list, daily_bar, trd_mtch_cfg):
    trade_info = pd.concat((bod_share, trade_list), axis=1).fillna(0.0)
    trade_info = pd.concat((trade_info, daily_bar.loc[trade_info.index]), axis=1)
    #
    to_sel_port = trade_info[(trade_info["trade_share"] < -SHARE_EPSILON) | (trade_info["expected_share"] < SHARE_EPSILON)].copy()
    if trd_mtch_cfg["is_trade_limited"] == "on":
        to_sel_port = to_sel_port[((to_sel_port["volume"] > SHARE_EPSILON) & (to_sel_port["is_limit_eod"] > -0.5)) |
                                  (to_sel_port["remove_by_delist"] > 0.5)].copy()
        to_sel_port["trade_share"].mask(to_sel_port["hit_low_limit"] > 0.5,
                                        to_sel_port["trade_share"] * trd_mtch_cfg["limit_trade_ratio"],
                                        inplace=True)
    #
    to_buy_port = trade_info[trade_info["trade_share"] > SHARE_EPSILON].copy()
    if trd_mtch_cfg["is_trade_limited"] == "on":
        to_buy_port = to_buy_port[(to_buy_port["volume"] > SHARE_EPSILON) & (to_buy_port["is_limit_eod"] < 0.5)].copy()
        to_buy_port["trade_share"].mask(to_buy_port["hit_up_limit"] > 0.5,
                                        to_buy_port["trade_share"] * trd_mtch_cfg["limit_trade_ratio"],
                                        inplace=True)
    #
    price_type = trd_mtch_cfg["price_type"]
    cash = bod_cash - (to_sel_port[price_type] * to_sel_port['trade_share']).sum()
    if to_buy_port.empty:
        buy_ratio = 0.0
    else:
        buy_ratio = cash / (to_buy_port[price_type] * to_buy_port['trade_share']).sum()
    #
    buy_ratio = max(0.0, buy_ratio)
    cash = cash - buy_ratio * (to_buy_port[price_type] * to_buy_port['trade_share']).sum()
    traded_cash = cash - bod_cash
    #
    traded_list = pd.concat((
        to_sel_port.assign(traded_share=to_sel_port["trade_share"]),
        to_buy_port.assign(traded_share=to_buy_port['trade_share'] * buy_ratio)
    ), axis=0)
    traded_list = traded_list[["traded_share", price_type]].rename(columns={price_type: "traded_price"}, errors="raise")
    return traded_list, traded_cash


def _account(bod_share, bod_cash, traded_list, traded_cash, accnt_cfg):
    eod_share = traded_list["traded_share"].add(bod_share, fill_value=0)
    eod_share = eod_share[eod_share > EPSILON].rename("eod_share")
    #
    sold_cash = traded_list.loc[traded_list["traded_share"] < -EPSILON,
                                ["traded_share", "traded_price"]].product(axis=1).abs().sum()
    bought_cash = traded_list.loc[traded_list["traded_share"] > EPSILON,
                                  ["traded_share", "traded_price"]].product(axis=1).abs().sum()
    trading_cost = sold_cash * accnt_cfg["stamp_tax"] + (sold_cash + bought_cash) * accnt_cfg["trading_fee"]
    eod_cash = bod_cash + traded_cash - trading_cost
    return eod_share, eod_cash, trading_cost


def _make_to_trade_info(eod_share, eod_cash, eod_price, expected_weight, trd_make_cfg):
    trade_info = pd.concat((eod_share, expected_weight, eod_price), axis=1)
    trade_info["eod_share"].fillna(0.0, inplace=True)
    trade_info["expected_weight"].fillna(0.0, inplace=True)
    trade_info = trade_info[(trade_info["eod_share"] > EPSILON) | (trade_info["expected_weight"] > EPSILON)].copy()
    #
    nav = (trade_info["eod_share"] * trade_info["eod_price"]).sum() + eod_cash
    trade_info["expected_share"] = nav * trade_info["expected_weight"] / trade_info["eod_price"]
    #
    trade_info["trade_share"] = trade_info["expected_share"] - trade_info["eod_share"]
    return trade_info


def calc_portfolio_eod_info(root_path, init_cash, port_daily_weight, trd_mtch_cfg, accnt_cfg, trd_make_cfg):
    assert pd.Index(['WeightDate', 'Code', 'expected_weight']).difference(port_daily_weight.columns).empty
    assert port_daily_weight['WeightDate'].is_monotonic_increasing
    all_data = _prepare_all_data(root_path, port_daily_weight)
    # init
    eod_cash = init_cash
    eod_share = pd.Series(name="eod_share", dtype="object")
    trade_list = pd.DataFrame(columns=["trade_share", "expected_share"])
    #
    date_list = CALENDAR_UTIL.get_ranged_trading_dates(
        all_data.index.get_level_values('CalcDate')[0], all_data.index.get_level_values('CalcDate')[-1])
    eod_info_data = list()
    cash_account = list()
    for date in date_list:
        stk_data = all_data.loc[date].set_index(["Code"])
        divnsplit = stk_data[["split_ratio", "div_rate"]].copy()
        daily_bar = stk_data.drop(columns=["expected_weight", "split_ratio", "div_rate"])
        eod_price = daily_bar["close_price"].rename("eod_price").copy()
        #
        bod_share, bod_cash, trade_list = _calc_bod_port(eod_share, eod_cash, trade_list, divnsplit)
        traded_list, traded_cash = _make_match(bod_share, bod_cash, trade_list, daily_bar, trd_mtch_cfg)
        eod_share, eod_cash, trading_cost = _account(bod_share, bod_cash, traded_list, traded_cash, accnt_cfg)
        trade_info = _make_to_trade_info(eod_share, eod_cash, eod_price, stk_data["expected_weight"], trd_make_cfg)
        trade_list = trade_info[["trade_share", "expected_share"]].copy()
        #
        eod_info = pd.concat((trade_info[["trade_share", "expected_share", "eod_share"]], traded_list), axis=1)
        eod_info = pd.concat((eod_info, eod_price[eod_info.index]), axis=1)
        eod_info["trade_share"].fillna(0.0, inplace=True)
        eod_info["expected_share"].fillna(0.0, inplace=True)
        eod_info["eod_share"].fillna(0.0, inplace=True)
        eod_info["traded_share"].fillna(0.0, inplace=True)
        eod_info.index = pd.MultiIndex.from_product([[date], eod_info.index], names=["CalcDate", "Code"])
        cash_account.append([date, eod_cash, trading_cost])
        eod_info_data.append(eod_info)
    cash_account = pd.DataFrame(cash_account, columns=["CalcDate", "eod_cash", "trading_cost"]).set_index("CalcDate")
    eod_info_data = pd.concat(eod_info_data, axis=0).reset_index(drop=False)
    return eod_info_data, cash_account