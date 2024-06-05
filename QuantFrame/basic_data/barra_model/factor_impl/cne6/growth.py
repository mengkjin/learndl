from .util import get_latest_year_rptprd_range, prepare_latest_3_years_finc_data
from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd
import numpy as np
from basic_src_data.wind_tools.finance import load_income_by_wind
from .cne6_utils import apply_qtile_shrink, apply_zscore


SALES_EPSILON = 1e7


def _calc_sales_grw(data):
    sale_data = data.set_index(["AnnDate", "Code", "report_period"])[[
        "revenue", "revenue_last_year", "revenue_last_2_year"]].dropna(how="any")
    sale_data = sale_data[sale_data["revenue"] >= SALES_EPSILON].copy()
    x = np.array([1, 2, 3])
    beta = sale_data.sub(sale_data.mean(axis=1), axis=0).dot(x - x.mean()) / (x - x.mean()).dot(x - x.mean())
    rtn = beta / sale_data.mean(axis=1)
    rtn.rename("sgro", inplace=True)
    return rtn


def _calc_earnings_grw(data):
    np_data = data.set_index(["AnnDate", "Code", "report_period"])[[
        "net_profit", "net_profit_last_year", "net_profit_last_2_year"]].dropna(how="any")
    x = np.array([1, 2, 3])
    beta = np_data.sub(np_data.mean(axis=1), axis=0).dot(x - x.mean()) / (x - x.mean()).dot(x - x.mean())
    rtn = beta / np_data.abs().mean(axis=1)
    rtn.rename("egro", inplace=True)
    return rtn


def _fwd_fill_finc_features(features, ecd):
    assert pd.Index(["CalcDate", "Code", "report_period"]).difference(features.columns).empty
    assert features["CalcDate"].is_monotonic_increasing
    date_range = CALENDAR_UTIL.get_ranged_dates(features["CalcDate"].min(), ecd)
    rtn = features.set_index(["CalcDate", "Code"]).unstack()\
        .reindex(date_range).fillna(method="ffill").stack().reset_index(drop=False)
    # create a CalcDate DateFrame to reduce calculation
    calc_date_alias = rtn[["CalcDate"]].drop_duplicates()
    calc_date_alias["min_prd_rpd"] = calc_date_alias["CalcDate"].apply(
        lambda x: get_latest_year_rptprd_range(x)[0]).astype(str)
    rtn = pd.merge(rtn, calc_date_alias, on=["CalcDate"], how="left"
                   ).query("report_period >= min_prd_rpd").drop(columns=["min_prd_rpd", "report_period"])
    return rtn


def calc_growth(scd, ecd):
    FWD_FILL_WINSIZE = 500
    calc_sd = CALENDAR_UTIL.get_latest_n_dates(scd, FWD_FILL_WINSIZE + 1)[0]
    data_sd = CALENDAR_UTIL.get_n_years_before([calc_sd], 2)[0]
    finc_data = load_income_by_wind(data_sd, ecd, ["net_profit", "revenue"])
    finc_data = finc_data[finc_data['Code'].str[-2:] != 'BJ'].copy()
    #
    all_data = prepare_latest_3_years_finc_data(finc_data)
    all_data = all_data[all_data["AnnDate"] >= calc_sd].copy()
    #
    egro = _calc_earnings_grw(all_data)
    sgro = _calc_sales_grw(all_data)
    rtn = pd.concat((egro, sgro), axis=1, sort=True).reset_index(drop=False)
    rtn.rename(columns={"AnnDate": "CalcDate"}, errors="raise", inplace=True)
    #
    rtn = _fwd_fill_finc_features(rtn, ecd)
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both"), ["CalcDate", "Code", "egro", "sgro"]].set_index(['CalcDate', 'Code'])
    rtn = apply_qtile_shrink(rtn)
    rtn = apply_zscore(rtn).mean(axis=1, skipna=False)
    rtn = apply_zscore(rtn).rename('Growth')
    return rtn
