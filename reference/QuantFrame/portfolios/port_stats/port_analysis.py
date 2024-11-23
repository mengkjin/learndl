from port_generals.weight_utils import get_relative_port_stk_wt
from port_stats.ts.risk_loading import calc_port_risk_loading
import pandas as pd
from daily_bar.api import load_daily_bar_data
from events_system.calendar_util import CALENDAR_UTIL
from stk_index_utils.api import load_index_level
import numpy as np


def _estimate_port_tracking_error(root_path, port_weight, bm_index, lookback_winsize):
    ecd = port_weight["CalcDate"].iloc[0]
    data_sd = CALENDAR_UTIL.get_last_trading_dates([ecd], inc_self_if_is_trdday=True, n=lookback_winsize)[0]
    ret_data = load_daily_bar_data(root_path, "basic", data_sd, ecd)
    #
    ret_data = ret_data.set_index(["CalcDate", "Code"])["ret"].unstack()[port_weight["Code"].tolist()].copy()
    weight_se = port_weight.set_index(["Code"])["target_weight"].copy()
    port_ret = ret_data.dot(weight_se)
    #
    index_level = load_index_level(root_path, data_sd, ecd, bm_index).set_index(["CalcDate"])
    index_ret = index_level["close_level"] / index_level["preclose_level"] - 1.0
    #
    rtn = (port_ret - index_ret).std() * np.sqrt(250)
    return rtn


def analyse_port_risk(root_path, port_weight, bm_index, barra_type, lookback_winsize):
    assert pd.Index(['CalcDate', 'Code', 'target_weight']).difference(port_weight.columns).empty
    port_df = port_weight[["CalcDate", "Code", "target_weight"]].rename(
        columns={"CalcDate": "WeightDate", "target_weight": "weight"}, errors="raise")
    rel_port_weight = get_relative_port_stk_wt(root_path, port_df, bm_index)
    constituent_df = rel_port_weight[rel_port_weight['index_weight'] > 0]
    constituent_ratio = constituent_df.groupby('WeightDate')['weight'].sum().reset_index()
    constituent_ratio.set_index('WeightDate', inplace=True)
    rel_port_weight = rel_port_weight[['WeightDate', 'Code', 'rel_weight']].rename(columns={"rel_weight": "weight"}, errors="raise")
    #
    risk_loading = calc_port_risk_loading(root_path, rel_port_weight[["WeightDate", "Code", "weight"]].rename(
        columns={"WeightDate": "CalcDate"}, errors="raise"), barra_type)
    risk_loading.set_index(["CalcDate"], inplace=True)
    style_bias = risk_loading.filter(regex="STYLE", axis=1)
    style_bias.columns = style_bias.columns.str.replace("STYLE.", "", regex=False)
    #
    industry_bias = risk_loading.filter(regex="INDUSTRY", axis=1)
    industry_bias.columns = industry_bias.columns.str.replace("INDUSTRY.", "", regex=False)
    #
    calc_date_list = sorted(list(port_weight["CalcDate"].unique()))
    tracking_error = dict()
    for date in calc_date_list:
        day_port_weight = port_weight[port_weight["CalcDate"] == date].copy()
        tracking_error[date] = _estimate_port_tracking_error(root_path, day_port_weight, bm_index, lookback_winsize)
    tracking_error = pd.Series(tracking_error).to_frame(name="tracking_error")
    return style_bias, industry_bias, tracking_error, constituent_ratio
