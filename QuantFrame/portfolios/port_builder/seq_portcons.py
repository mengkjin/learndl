import pandas as pd
from .risk_cond_builder import get_general_conditions, get_cov_mat
from daily_bar.api import load_daily_bar_data
from events_system.calendar_util import CALENDAR_UTIL
from .port_con_center import con_port


def prepare_data_1_day(fcst_df, all_conditions, calc_date, eod_port):
    rtn = pd.merge(fcst_df[fcst_df['CalcDate'] == calc_date], eod_port[['Code', 'weight']].rename(columns={'weight': 'w0'}), how='outer', on=['Code'])
    rtn['fcst'].fillna(0.0, inplace=True)  # comment: code in eod_port but not in fcst_df will be kept with zero fcst
    rtn['w0'].fillna(0.0, inplace=True)
    rtn = pd.merge(rtn, all_conditions[all_conditions['CalcDate'] == calc_date], how='inner', on=['CalcDate', 'Code'])
    return rtn


def trade_and_match(expected_target_port, cash, ret):
    expected_target_port['price'] = 1.0 + ret.to_numpy()
    to_kep_port = expected_target_port[expected_target_port['delta_weight'].abs() < 0.0001].copy()
    to_sel_port = expected_target_port[expected_target_port['delta_weight'] <= -0.0001].copy()
    to_buy_port = expected_target_port[expected_target_port['delta_weight'] >= 0.0001].copy()
    cash = cash + -(to_sel_port['price'] * to_sel_port['delta_weight']).sum()
    if to_buy_port.empty:
        buy_ratio = 0.0
    else:
        buy_ratio = cash / (to_buy_port['price'] * to_buy_port['delta_weight']).sum()
    cash = cash - buy_ratio * (to_buy_port['price'] * to_buy_port['delta_weight']).sum()
    eod_port = pd.concat((
        to_kep_port[['Code']].assign(eod_amount=to_kep_port['target_weight'] * to_kep_port['price']),
        to_sel_port[['Code']].assign(eod_amount=to_sel_port['target_weight'] * to_sel_port['price']),
        to_buy_port[['Code']].assign(
            eod_amount=(to_buy_port['target_weight'] - to_buy_port['delta_weight'] * (1. - buy_ratio)) *
                       to_buy_port['price'])
    ))
    eod_port = eod_port.assign(weight=eod_port['eod_amount'] / eod_port['eod_amount'].sum())[['Code', 'weight']].copy()
    return eod_port, cash


def _calc_target_weights_impl_loop(port_date_timeline, fcst_df, all_conditions, risk_cov_mats, stkret, port_con, initial_weight):
    port_date_timeline = port_date_timeline.set_index(['CalcDate']).reindex(
        CALENDAR_UTIL.get_ranged_trading_dates(
            port_date_timeline['CalcDate'].iloc[0], port_date_timeline['CalcDate'].iloc[-1])).fillna('hld')
    port_date_timeline = port_date_timeline.reset_index().values.tolist()
    rtn = list()
    stkret["CNY"] = 0.0
    expected_target_port = pd.DataFrame(columns=['CalcDate', 'TradeDate', 'Code', 'delta_weight', 'target_weight'])
    cash = 1.0
    assert port_date_timeline[0][1] != 'hld'
    for idx, (date, date_type) in enumerate(port_date_timeline):
        ret = stkret.loc[date, expected_target_port['Code']].fillna(0.0)
        if idx == 0 and initial_weight is not None:
            eod_port = initial_weight[["Code", "weight"]].copy()
            cash = 0.0
        else:
            eod_port, cash = trade_and_match(expected_target_port, cash, ret)
        #
        trd_date = CALENDAR_UTIL.get_next_trading_dates([date], inc_self_if_is_trdday=False)[0]
        if date_type != 'hld':
            conditions = prepare_data_1_day(fcst_df, all_conditions, date, eod_port)
            expected_target_port = con_port(conditions, risk_cov_mats[date], 'fcst', port_con)
            port_position = expected_target_port["target_weight"].sum()
            #
            expected_target_port = expected_target_port.loc[expected_target_port['target_weight'] > 0.0001, ['Code', 'target_weight']].copy()
            expected_target_port["target_weight"] = expected_target_port["target_weight"] / expected_target_port["target_weight"].sum() * port_position
            expected_target_port = expected_target_port[["Code", "target_weight"]].copy()
            #
            cash_df = pd.DataFrame([["CNY", 1 - expected_target_port["target_weight"].sum()]], columns=["Code", "target_weight"])
            expected_target_port = pd.concat((expected_target_port, cash_df), axis=0).sort_values(["Code"])
            #
            expected_target_port = pd.merge(expected_target_port, eod_port[eod_port['weight'] > 0.0001].rename(columns={'weight': 'last_weight'}), how='outer', on=['Code'])
            expected_target_port = expected_target_port.fillna(0.0).assign(CalcDate=date, TradeDate=trd_date)
            expected_target_port['delta_weight'] = expected_target_port['target_weight'] - expected_target_port['last_weight']
            if (expected_target_port['delta_weight'].abs() < 0.0001).all():
                expected_target_port['target_weight'] = expected_target_port['last_weight']
                expected_target_port['delta_weight'] = 0.0
                expected_target_port = expected_target_port[['CalcDate', 'TradeDate', 'Code', 'delta_weight', 'target_weight']].copy()
            else:
                expected_target_port = expected_target_port[['CalcDate', 'TradeDate', 'Code', 'delta_weight', 'target_weight']].copy()
                rtn.append(expected_target_port)
        else:
            expected_target_port = eod_port.assign(CalcDate=date, TradeDate=trd_date, delta_weight=0.0)[['CalcDate', 'TradeDate', 'Code', 'delta_weight', 'weight']].rename(columns={'weight': 'target_weight'})
    rtn = pd.concat(rtn, axis=0)[['CalcDate', 'TradeDate', 'Code', 'target_weight']].copy()
    rtn = rtn[rtn["Code"] != "CNY"].copy()
    return rtn


def prepare_data(root_path, calc_date_list, stk_pool_name, bm_index, risk_mdl_name):
    trade_date_list = CALENDAR_UTIL.get_next_trading_dates(calc_date_list, inc_self_if_is_trdday=False)
    cov_conditions = get_cov_mat(root_path, risk_mdl_name, calc_date_list)
    assert sorted(cov_conditions.keys()) == calc_date_list
    all_conditions = get_general_conditions(root_path, stk_pool_name, risk_mdl_name, bm_index, calc_date_list)
    assert all_conditions['CalcDate'].drop_duplicates().tolist() == calc_date_list
    all_conditions = pd.merge(
        all_conditions,
        pd.DataFrame(zip(calc_date_list, trade_date_list), columns=['CalcDate', 'TradeDate']),
        how='left', on=['CalcDate'])
    industry_cols = all_conditions.columns[all_conditions.columns.str.contains('INDUSTRY\.')]
    all_conditions[industry_cols] = all_conditions[industry_cols].fillna(0.0)
    all_conditions.rename(columns={"member_weight": 'bm_index'}, errors="raise", inplace=True)
    return all_conditions, cov_conditions


def get_calc_dates(date_strategy, scd, ecd, fcst_df):
    freq = date_strategy['freq']
    if freq == 'week':
        calc_date_list = CALENDAR_UTIL.get_ranged_eow_trading_dates(
            CALENDAR_UTIL.get_latest_n_eow_trading_dates(scd, 1)[0],
            ecd
        )
    elif freq == 'month':
        calc_date_list = CALENDAR_UTIL.get_ranged_eom_trading_dates(
            CALENDAR_UTIL.get_latest_n_eom_trading_dates(scd, 1)[0],
            ecd
        )
    elif freq == 'adapt_to_fcst':
        calc_date_list = fcst_df["CalcDate"].unique().tolist()
    else:
        assert False
    rtn = list(zip(calc_date_list, ['cal'] * len(calc_date_list)))
    rtn = sorted(rtn)
    rtn = pd.DataFrame(rtn, columns=['CalcDate', 'DateType'])
    return rtn


def calc_target_weights(root_path, fcst_df, date_strategy, env_config, strategy_configs, scd, ecd, initial_weight=None):
    assert fcst_df["CalcDate"].is_monotonic_increasing
    port_date_timeline = get_calc_dates(date_strategy, scd, ecd, fcst_df)
    assert set(port_date_timeline['CalcDate']).issubset(set(fcst_df["CalcDate"]))
    fcst_df = pd.merge(fcst_df, port_date_timeline[['CalcDate']], how='right', on='CalcDate')
    assert fcst_df.notnull().all().all()
    all_data, cov_mats = prepare_data(
        root_path, fcst_df['CalcDate'].drop_duplicates().tolist(), env_config["stock_universe"],
        env_config["bm_index"], env_config["risk_model_nm"])
    all_data['rsk_bm'] = all_data['bm_index']
    stk_ret = load_daily_bar_data(root_path, 'basic', port_date_timeline['CalcDate'].iloc[0], port_date_timeline['CalcDate'].iloc[-1]) \
        .set_index(['CalcDate', 'Code'])['ret'].unstack()
    rtn = dict()
    for strategy_ticker, port_con_info in strategy_configs.items():
        df = _calc_target_weights_impl_loop(port_date_timeline, fcst_df, all_data, cov_mats, stk_ret, port_con_info, initial_weight)
        df.sort_values(['CalcDate', 'TradeDate', 'Code'], inplace=True)
        rtn[strategy_ticker] = df[df['CalcDate'].between(scd, ecd)].copy()
        print("  status::fm_portfolios>>{0} strategy is calculated.".format(strategy_ticker))
    return rtn