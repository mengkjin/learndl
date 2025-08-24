import numpy as np
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from basic_src_data.wind_tools.basic import get_listed_ashare_codes_from_winddf
from daily_bar.api import load_daily_bar_data
from daily_kbar_util.limits_price import calc_is_limited


def _query_data(root_path, start_date_, end_date_):
    trading_data = load_daily_bar_data(root_path, "basic", start_date_, end_date_)
    trading_data = trading_data.loc[trading_data['Code'].str[-2:] != 'BJ', ['CalcDate', 'Code', 'close_price', 'prev_close', 'high_price', 'low_price', 'amount']].copy()
    is_limited = calc_is_limited(trading_data, ['is_open_limit', 'is_limit_eod'])
    # trading_data[['is_open_limit', 'is_limit_eod']] = is_limited[['is_open_limit', 'is_limit_eod']]
    trading_data['is_limit_eod'] = is_limited['is_limit_eod']
    cap_data = load_daily_bar_data(root_path, "valuation", start_date_, end_date_)
    cap_data = cap_data.loc[cap_data['Code'].str[-2:] != 'BJ', ['CalcDate', 'Code', 'total_value']].copy()
    rtn = pd.merge(trading_data, cap_data, how="inner", on=['CalcDate', 'Code'], sort=True)
    # rtn['amount'] = rtn['amount'].mask((rtn['amount'] < 0.01) | (rtn['is_open_limit'].abs() > 0.5), np.nan)
    rtn['amount'] = rtn['amount'].mask((rtn['amount'] < 0.01), np.nan)
    rtn = rtn[['CalcDate', 'Code', 'amount', 'close_price', 'total_value']].copy()
    return rtn


def get_trading_data_filter_impl(root_path, start_date_, end_date_):
    hist_trading_data = _query_data(root_path, str(int(start_date_[:4]) - 1) + start_date_[4:], end_date_)
    hist_trading_data = hist_trading_data.set_index(['CalcDate', 'Code']).unstack()
    ma_amt = hist_trading_data['amount'].rolling(240, min_periods=120).mean().stack().rename('ma_amt')
    ma_cls = hist_trading_data['close_price'].rolling(20, min_periods=1).mean().stack().rename('ma_clsprc')
    ma_cap = hist_trading_data['total_value'].rolling(60, min_periods=1).mean().stack().rename('ma_ttlcap')
    trade_info = pd.concat((ma_amt, ma_cls, ma_cap), axis=1, sort=True)
    listed_stocks = get_listed_ashare_codes_from_winddf(start_date_, end_date_)
    listed_stocks = listed_stocks[
        listed_stocks['Date'].isin(trade_info.index.get_level_values('CalcDate').drop_duplicates())].copy()
    listed_stocks = pd.MultiIndex.from_frame(listed_stocks, names=['CalcDate', 'Code'])
    trade_info = trade_info.reindex(listed_stocks)
    trade_info[['ma_amt_pctile', 'ma_ttlcap_pctile']] = trade_info.groupby('CalcDate')[['ma_amt', 'ma_ttlcap']].rank(pct=True)
    trade_info.reset_index(inplace=True)
    df = trade_info.loc[trade_info['CalcDate'].between(start_date_, end_date_)].copy()
    return df


def get_trading_data_filter(root_path, start_date, end_date):
    data_start_date = CALENDAR_UTIL.get_last_trading_dates([start_date], inc_self_if_is_trdday=True)[0]
    df = get_trading_data_filter_impl(root_path, data_start_date, end_date)
    exists_dates = df['CalcDate'].drop_duplicates()
    query_dates = CALENDAR_UTIL.get_ranged_dates(start_date, end_date)
    missed_dates = sorted(list(set(query_dates).difference(set(exists_dates))))
    latest_dates = [exists_dates[exists_dates < d].max() for d in missed_dates]
    date_maps = pd.DataFrame([missed_dates, latest_dates], index=['MissedDate', 'CalcDate']).T
    assert date_maps['CalcDate'].notna().all()
    missed_df = pd.merge(date_maps, df, how='left', on=['CalcDate']).drop(columns=['CalcDate']).rename(
        columns={'MissedDate': 'CalcDate'})
    rtn = pd.concat((df, missed_df), axis=0).sort_values(['CalcDate', 'Code'])
    rtn = rtn[rtn['CalcDate'].between(start_date, end_date)].reset_index(drop=True)
    return rtn

