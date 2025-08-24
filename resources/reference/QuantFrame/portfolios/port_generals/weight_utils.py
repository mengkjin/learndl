from daily_bar.api import load_daily_bar_data
from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd
from port_generals.to_patch_delist_earlier import get_early_delist


EPSILON = 1e-10
WEIGHT_EPSILON = 1e-5
AMOUNT_EPSILON = 1e-4


def _patch_delist_stocks(target_port, early_delist_stks, date):
    if date in early_delist_stks.index:
        to_delist_codes = early_delist_stks.loc[[date], "Code"].tolist()
        port_delist_codes = list(set(target_port["Code"]) & set(to_delist_codes))
        if port_delist_codes:
            delist_flg = target_port["Code"].isin(port_delist_codes)
            target_port.loc[delist_flg, 'target_weight'] = 0.0
            target_port.loc[delist_flg, 'delta_share'] = - target_port.loc[delist_flg, 'weight']
    return target_port


def calc_port_eod_weight(root_path, input_target_port, ecd):
    assert pd.Index(["TradeDate", "CalcDate", "Code", "target_weight"]).difference(input_target_port.columns).empty
    assert input_target_port["TradeDate"].unique().tolist() == \
           CALENDAR_UTIL.get_next_trading_dates(input_target_port["CalcDate"].unique().tolist(), inc_self_if_is_trdday=False)
    assert set(input_target_port['CalcDate']).issubset(
        CALENDAR_UTIL.get_ranged_trading_dates(input_target_port['CalcDate'].min(), input_target_port['CalcDate'].max()))
    assert (input_target_port.groupby(["CalcDate"])["target_weight"].sum() <= 1 + WEIGHT_EPSILON).all()
    first_date = input_target_port['CalcDate'].iloc[0]
    daily_bar = load_daily_bar_data(root_path, "basic", first_date, ecd)
    stk_ret = daily_bar.set_index(["CalcDate", "Code"])["ret"].unstack()
    stk_ret["CNY"] = 0.0
    early_delist_stks = get_early_delist(root_path, first_date, ecd).set_index(["Date"])
    #
    target_port = pd.DataFrame(columns=["CalcDate", "Code", "delta_share", "target_weight"])
    date_list = CALENDAR_UTIL.get_ranged_trading_dates(first_date, ecd)
    rtn = list()
    cash = 1.0
    for date in date_list:
        # 1. simulate trade
        ret = stk_ret.loc[date, target_port['Code']].fillna(0.0)
        target_port['price'] = 1.0 + ret.to_numpy()
        to_kep_port = target_port[(target_port['delta_share'].abs() <= WEIGHT_EPSILON) &
                                  (target_port['target_weight'] > WEIGHT_EPSILON)].copy()
        to_sel_port = target_port[(target_port['delta_share'] < - WEIGHT_EPSILON) |
                                  (target_port['target_weight'] <= WEIGHT_EPSILON)].copy()
        to_buy_port = target_port[target_port['delta_share'] > WEIGHT_EPSILON].copy()
        #
        cash = cash + abs((to_sel_port['price'] * to_sel_port['delta_share']).sum())
        if (to_buy_port['price'] * to_buy_port['delta_share']).sum() < AMOUNT_EPSILON:
            buy_ratio = 0.0
        else:
            buy_ratio = cash / (to_buy_port['price'] * to_buy_port['delta_share']).sum()
        cash = cash - buy_ratio * (to_buy_port['price'] * to_buy_port['delta_share']).sum()
        eod_port = pd.concat((
            to_kep_port[['Code']].assign(eod_amount=to_kep_port['target_weight'] * to_kep_port['price']),
            to_sel_port[['Code']].assign(eod_amount=to_sel_port['target_weight'] * to_sel_port['price']),
            to_buy_port[['Code']].assign(
                eod_amount=(to_buy_port['target_weight'] - to_buy_port['delta_share'] * (1. - buy_ratio)) * to_buy_port['price'])
        ))
        eod_port = eod_port.assign(weight=eod_port['eod_amount'] / eod_port['eod_amount'].sum(), CalcDate=date)
        eod_port = eod_port.loc[eod_port['weight'] > EPSILON, ['CalcDate', 'Code', 'weight']].copy()
        # 2. after 16:00 prepare tomorrow trading
        target_port = input_target_port[input_target_port['CalcDate'] == date].copy()
        if not target_port.empty:
            target_port = pd.merge(target_port[['CalcDate', 'Code', 'target_weight']], eod_port, how='outer', on=['Code'])
            target_port = target_port.fillna(0.0).assign(CalcDate=date)
            target_port['delta_share'] = target_port['target_weight'] - target_port['weight']
            target_port = _patch_delist_stocks(target_port, early_delist_stks, date)
            target_port = target_port[['CalcDate', 'Code', 'delta_share', 'target_weight']].copy()
        else:
            target_port = eod_port[['CalcDate', 'Code', 'weight']].assign(delta_share=0.0, target_weight=eod_port['weight'])
            target_port = _patch_delist_stocks(target_port, early_delist_stks, date)
            target_port.drop(columns=["weight"], inplace=True)
        #
        eod_info = pd.merge(eod_port[["CalcDate", "Code", "weight"]], target_port[["CalcDate", "Code", "target_weight"]],
                            how="outer", on=["CalcDate", "Code"])
        eod_info[["weight", "target_weight"]] = eod_info[["weight", "target_weight"]].fillna(0.0)
        rtn.append(eod_info)
    rtn = pd.concat(rtn, axis=0)
    rtn = rtn[['CalcDate', 'Code', 'weight', 'target_weight']].rename(
        columns={"CalcDate": "WeightDate", "target_weight": "expected_weight"}, errors="raise")
    return rtn


from stk_index_utils.api import load_index_weight_data


def get_relative_port_stk_wt(root_path, port_stk, bm_index, back_shift_n=0):
    assert port_stk.columns.equals(pd.Index(['WeightDate', 'Code', 'weight']))
    port_stk = port_stk.sort_values(["WeightDate", "Code"])
    #
    weight_date_list = port_stk["WeightDate"].unique().tolist()
    assert all([CALENDAR_UTIL.is_it('is_trading', d) for d in weight_date_list])
    bm_date_list = CALENDAR_UTIL.get_last_trading_dates(weight_date_list, inc_self_if_is_trdday=True, raise_error_if_nan=True, n=1 + back_shift_n)
    bm_index_df = load_index_weight_data(root_path, bm_date_list[0], bm_date_list[-1], bm_index)
    bm_index_df.rename(columns={"member_weight": "index_weight"}, errors="raise", inplace=True)
    bm_index_df = bm_index_df[bm_index_df["index_weight"] > 1e-8].copy()
    bm_index_df = pd.merge(pd.DataFrame(zip(weight_date_list, bm_date_list), columns=['WeightDate', "CalcDate"]), bm_index_df,
                           how="left", on=["CalcDate"])
    rtn = pd.merge(port_stk, bm_index_df[['WeightDate', 'Code', "index_weight"]], how="outer", on=["WeightDate", "Code"])
    rtn['weight'] = rtn['weight'].fillna(0.0)
    rtn["rel_weight"] = rtn["weight"] - rtn["index_weight"].fillna(0.0)
    rtn = rtn[['WeightDate', 'Code', 'rel_weight', 'weight', 'index_weight']].sort_values(["WeightDate", "Code"])
    return rtn