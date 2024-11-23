import pandas as pd
import numpy as np
from basic_src_data.wind_tools.daily_bar import load_daily_bar_by_wind
from basic_src_data.wind_tools.valuation import load_valuation_by_wind
from events_system.calendar_util import CALENDAR_UTIL
import os


def _patch_basic_data(data):
    assert data["CalcDate"].is_monotonic_increasing
    scd, ecd = data["CalcDate"].iloc[0], data["CalcDate"].iloc[-1]
    patch_sd, patch_ed = "2020-10-29", "2022-12-01"
    if not (patch_sd > ecd or patch_ed < scd):
        patch_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'basic_data_for_patch.csv'))
        patch_data = patch_data.loc[patch_data["CalcDate"].between(scd, ecd, inclusive="both"), data.columns].copy()
        rtn = pd.concat((data, patch_data), axis=0).sort_values(["CalcDate", "Code"]).\
            drop_duplicates(subset=["CalcDate", "Code"], keep="first").reset_index(drop=True)
    else:
        rtn = data
    return rtn


def _patch_valuation_data(data):
    assert data["CalcDate"].is_monotonic_increasing
    scd, ecd = data["CalcDate"].iloc[0], data["CalcDate"].iloc[-1]
    patch_sd, patch_ed = "2020-10-29", "2022-11-30"
    if not (patch_sd > ecd or patch_ed < scd):
        patch_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'val_data_for_patch.csv'))
        patch_data = patch_data.loc[patch_data["CalcDate"].between(scd, ecd, inclusive="both"), data.columns].copy()
        rtn = pd.concat((data, patch_data), axis=0).sort_values(["CalcDate", "Code"]).\
            drop_duplicates(subset=["CalcDate", "Code"], keep="first").reset_index(drop=True)
    else:
        rtn = data
    return rtn


def calc_basic_data(start_calc_date_, end_calc_date_):
    flds = ['open_price', 'close_price', 'high_price', 'low_price', 'prev_close', 'vwap', 'amount', 'volume', 'turnover',]
    wind_scd = CALENDAR_UTIL.get_latest_n_trading_dates(start_calc_date_, 10)[0]
    data = load_daily_bar_by_wind(wind_scd, end_calc_date_, flds)
    data = _patch_basic_data(data)
    data['is_traded'] = (data['volume'] > 1) * 1
    data['ret'] = data['close_price'] / data['prev_close'] - 1.0
    data['log_ret'] = np.log(data['close_price']) - np.log(data['prev_close'])
    data.set_index(['CalcDate', 'Code'], inplace=True)
    #
    ret_yesterday = np.log(data['close_price'] / data['vwap']).unstack().shift(1)
    ret_today = np.log(data['vwap'] / data['prev_close']).unstack()
    vwap_ret = (ret_today + ret_yesterday).stack().rename('vwp_log_ret')
    data['vwp_log_ret'] = vwap_ret
    data.reset_index(inplace=True)
    rtn = data.loc[data['CalcDate'].between(start_calc_date_, end_calc_date_),
                ['CalcDate', 'Code', 'is_traded', 'open_price', 'high_price', 'low_price', 'close_price', 'prev_close',
                 'volume', 'amount', 'turnover', 'vwap', 'ret', 'log_ret', 'vwp_log_ret']].copy()
    return rtn


def calc_valuation_data(scd, ecd):
    flds = ["total_value", 'float_value', 'book_value', 'total_share', 'float_share']
    data = load_valuation_by_wind(scd, ecd, flds)
    data = _patch_valuation_data(data)
    return data