from .bp import calc_bp
from .size import calc_size, process_size, calc_nonlinear_size
from .earnvar import calc_earning_var
from .earnyld import calc_earnyld
from .growth import calc_growth
from .invsqlty import calc_invsqlty
from .leverage import calc_leverage
from .industry import calc_industry
from .liquidity import calc_liquidity
from .momentum import calc_mom
from .resvol import calc_resvol
from .profit import calc_profit
from .beta import calc_beta
from .divyld import calc_divyld
import pandas as pd
import numpy as np
from .cne6_utils import fill_nan
import matplotlib.pyplot as plt
from events_system.calendar_util import CALENDAR_UTIL
from .weights import calc_weight
from .company_property import calc_company_property


def transfer_data_to_calendar_data(data):
    data = data.rename(columns={"CalcDate": "trade_CalcDate"}, errors="raise")
    date_df = data[["trade_CalcDate"]].drop_duplicates()
    date_df["calendar_dates"] = date_df["trade_CalcDate"].apply(
        lambda x: CALENDAR_UTIL.get_ranged_dates(
            x, CALENDAR_UTIL.get_next_trading_dates([x], inc_self_if_is_trdday=False)[0])[:-1])
    rtn = pd.merge(data, date_df, on=["trade_CalcDate"], how="left").rename(
        columns={"calendar_dates": "CalcDate"}, errors="raise")
    rtn = rtn.explode("CalcDate").drop(columns=["trade_CalcDate"]).\
        sort_values(["CalcDate", "Code"])
    return rtn


def normalize(x, weight_name):
    to_nrm_col = x.columns.drop([weight_name])
    rtn = (x[to_nrm_col] - x[to_nrm_col].T.dot(x[weight_name]) / x[weight_name].sum()) / x[to_nrm_col].std()
    return rtn


def calc_barra_vals(root_path, scd, ecd):
    dscd = CALENDAR_UTIL.get_last_trading_dates([scd], inc_self_if_is_trdday=True)[0]
    industry = calc_industry(root_path, dscd, ecd).set_index(['CalcDate', 'Code'])
    #
    weights = calc_weight(root_path, dscd, ecd)
    assert weights.index.get_level_values('CalcDate').is_monotonic_increasing \
           and weights.index.get_level_values('CalcDate').drop_duplicates().tolist() == CALENDAR_UTIL.get_ranged_trading_dates(dscd, ecd)
    bp_val = calc_bp(root_path, dscd, ecd)
    size_val = calc_size(root_path, dscd, ecd)
    size_val = process_size(size_val, weights)
    cubic_size_val = calc_nonlinear_size(size_val, weights)
    profit = calc_profit(dscd, ecd)
    earnings_var = calc_earning_var(dscd, ecd)
    earnyild = calc_earnyld(root_path, dscd, ecd)
    growth = calc_growth(dscd, ecd)
    invsqlty = calc_invsqlty(dscd, ecd)
    leverage = calc_leverage(root_path, dscd, ecd)
    liquidity = calc_liquidity(root_path, dscd, ecd)
    momentum = calc_mom(root_path, dscd, ecd)
    resvol = calc_resvol(root_path, dscd, ecd)
    beta = calc_beta(root_path, dscd, ecd)
    divyld = calc_divyld(root_path, dscd, ecd)
    company_property = calc_company_property(dscd, ecd)
    #
    all_data = pd.concat((industry, bp_val, size_val, cubic_size_val, profit), axis=1, sort=True)
    del industry, bp_val, size_val, cubic_size_val, profit
    all_data = pd.concat((all_data, earnings_var, earnyild, growth, invsqlty), axis=1, sort=True)
    del earnings_var, earnyild, growth, invsqlty
    all_data = pd.concat((all_data, leverage, liquidity, momentum, resvol, beta, divyld, company_property), axis=1, sort=True)
    del leverage, liquidity, momentum, resvol, beta, divyld, company_property
    #
    flg = all_data[['industry', 'Bp', 'Size']].notnull().all(axis=1)
    all_data = all_data[flg].copy()
    all_data = pd.merge(weights.index.to_frame(index=False), all_data, how='inner', on=['CalcDate', 'Code'])
    assert all_data['CalcDate'].drop_duplicates().tolist() == CALENDAR_UTIL.get_ranged_trading_dates(dscd, ecd)
    #
    all_data["Property_State"].fillna(0.0, inplace=True)
    all_data = fill_nan(all_data.set_index(['CalcDate', 'Code']))
    assert all_data.notnull().all().all()
    all_data = pd.merge(all_data, weights, how='left', on=['CalcDate', 'Code'])
    assert all_data['size_weight'].notnull().all()
    all_data = all_data.set_index('industry', append=True)
    all_data = all_data.groupby('CalcDate', as_index=False, group_keys=False).apply(normalize, weight_name='size_weight')
    all_data.columns = ['STYLE.{0}'.format(style_nm) for style_nm in all_data.columns]
    #
    rtn = all_data.reset_index(drop=False)
    rtn.rename(columns={'industry': 'INDUSTRY.citics_1'}, errors="raise", inplace=True)
    rtn = transfer_data_to_calendar_data(rtn)
    rtn = rtn[rtn["CalcDate"].between(scd, ecd, inclusive="both")].copy()
    assert (rtn['Code'].str[-2:] != 'BJ').all() and (rtn['Code'].str[0].isin(['6', '0', '3'])).all()
    assert rtn.notnull().all().all()
    assert rtn['CalcDate'].drop_duplicates().tolist() == CALENDAR_UTIL.get_ranged_dates(scd, ecd)
    rtn = rtn[pd.Index(['CalcDate', 'Code']).append(rtn.columns.drop(['CalcDate', 'Code']))].copy()
    return rtn