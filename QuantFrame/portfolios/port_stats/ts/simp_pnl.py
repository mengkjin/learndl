import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from daily_bar.api import load_daily_bar_data


def calc_pnl(root_path, port_weights):
    assert port_weights.columns.equals(pd.Index(['WeightDate', 'Code', 'weight']))
    port_weights = port_weights.rename(columns={'WeightDate': 'CalcDate'}, errors='raise')
    assert port_weights['CalcDate'].unique().tolist() == CALENDAR_UTIL.get_ranged_trading_dates(
        port_weights['CalcDate'].iloc[0], port_weights['CalcDate'].iloc[-1]
    )
    port_weights['next_date'] = CALENDAR_UTIL.get_next_trading_dates(port_weights['CalcDate'], inc_self_if_is_trdday=False)
    daily_bar_data = load_daily_bar_data(
        root_path, 'basic', port_weights['CalcDate'].iloc[0],
        port_weights['next_date'].iloc[-1])[['CalcDate', 'Code', 'prev_close', 'close_price']].copy()
    daily_bar_data['ret'] = daily_bar_data['close_price'] / daily_bar_data['prev_close'] - 1.0
    daily_bar_data.rename(columns={'CalcDate': 'next_date'}, inplace=True)
    port_weights = pd.merge(port_weights, daily_bar_data[['next_date', 'Code', 'ret']], how='left', on=['next_date', 'Code'])
    nan_ret_data = port_weights[port_weights['ret'].isna()].copy()
    if not nan_ret_data.empty:
        port_weights["ret"].fillna(0.0, inplace=True)
        print("error")
        print(nan_ret_data)
    pnl = port_weights.groupby('next_date').apply(lambda x: x['weight'].dot(x['ret'])).reset_index(drop=False, name='ret')
    pnl['pnl'] = (1.0 + pnl['ret']).cumprod()
    rtn = pnl[['next_date', 'pnl']].copy().rename(columns={'next_date': 'CalcDate'})
    return rtn


from stk_index_utils.api import load_index_level


def calc_relative_pnl(root_path, pnl, bm_index):
    index_level = load_index_level(root_path, pnl['CalcDate'].iloc[0], pnl['CalcDate'].iloc[-1], bm_index)
    rtn = pd.concat((pnl.set_index(['CalcDate']), index_level.set_index(['CalcDate'])), axis=1, sort=True)
    rtn['close_level'] = rtn['close_level'] / rtn['close_level'].iloc[0]
    return rtn


def calc_relative_perf(root_path, pnl, bm_index):
    index_level = load_index_level(root_path, pnl['CalcDate'].iloc[0], pnl['CalcDate'].iloc[-1], bm_index)
    all_info = pd.merge(pnl, index_level, how='left', on=['CalcDate'])
    all_info['year'] = all_info['CalcDate'].str[:4]
    all_info = pd.merge(
        all_info[['year', 'CalcDate']].groupby('year')['CalcDate'].max(),
        all_info,
        how='left',
        on=['CalcDate']
    ).drop(columns=['year'])
    rtn = all_info.set_index('CalcDate').pct_change()
    rtn['excess'] = rtn['nav'] - rtn['close_level']
    return rtn


from stk_ret_tools.stk_ret_calculator import calc_stk_period_ret
import numpy as np


def _calc_port_prd_ret_for_impl(root_path, port_df, ed):
    trade_date_list = port_df["TradeDate"].unique().tolist()
    y_start = [d for d in trade_date_list if d < ed]
    assert y_start
    date_df = pd.concat((
        pd.Series(y_start).rename("y_start"),
        pd.Series(y_start[1:] + [ed]).rename("y_end")), axis=1, sort=True)
    #
    port_df = pd.merge(port_df, date_df, how="inner", left_on=["TradeDate"], right_on=["y_start"], sort=True)
    stk_prd_ret = calc_stk_period_ret(root_path, port_df[["Code", "y_start", "y_end"]])
    port_df = pd.merge(port_df, stk_prd_ret, how="left", on=["Code", "y_start", "y_end"])
    port_df["prd_ret"] = np.exp(port_df["prd_lg_ret"]) - 1.0
    port_df["wt_ret"] = port_df["target_weight"] * port_df["prd_ret"]
    rtn = port_df.groupby(["TradeDate"])["wt_ret"].sum().rename("port_ret")
    return rtn


def calc_port_prd_ret(root_path, port_df, sd, ed):
    assert port_df.columns.equals(pd.Index(["TradeDate", "Code", "target_weight"]))
    assert port_df["TradeDate"].is_monotonic_increasing
    trade_date_list = port_df["TradeDate"].unique().tolist()
    assert set(trade_date_list).issubset(CALENDAR_UTIL.get_ranged_trading_dates(trade_date_list[0], trade_date_list[-1]))
    assert port_df["TradeDate"].iloc[0] <= sd
    trade_dt_cover_sd = [d for d in trade_date_list if d <= sd][-1]
    #
    if trade_dt_cover_sd != sd:
        all_port_df = port_df[port_df["TradeDate"] >= trade_dt_cover_sd].copy()
        all_prd_ret_se = _calc_port_prd_ret_for_impl(root_path, all_port_df, ed)
        start_port_df = port_df[port_df["TradeDate"].between(trade_dt_cover_sd, sd, inclusive="both")].copy()
        start_prd_ret_se = _calc_port_prd_ret_for_impl(root_path, start_port_df, sd)
        #
        rtn = all_prd_ret_se[sd:].copy()
        rtn[sd] = (all_prd_ret_se[trade_dt_cover_sd] + 1.0) / (start_prd_ret_se[trade_dt_cover_sd] + 1.0) - 1.0
        rtn = rtn.sort_index()
    else:
        rtn = _calc_port_prd_ret_for_impl(root_path, port_df, ed)
    return rtn