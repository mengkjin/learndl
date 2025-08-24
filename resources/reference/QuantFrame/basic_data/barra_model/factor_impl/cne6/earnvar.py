from .util import get_latest_year_rptprd_range, prepare_latest_3_years_finc_data
from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd
from basic_src_data.wind_tools.finance import load_income_by_wind, load_cashflow_by_wind
from .cne6_utils import apply_qtile_shrink, apply_zscore
import numpy as np

SALES_EPSILON = 1e7


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


def calc_earnings_sales_var(scd, ecd):
    FWD_FILL_WINSIZE = 500
    calc_sd = CALENDAR_UTIL.get_latest_n_dates(scd, FWD_FILL_WINSIZE + 1)[0]
    data_sd = CALENDAR_UTIL.get_n_years_before([calc_sd], 2)[0]
    finc_data = load_income_by_wind(data_sd, ecd, ["revenue", "net_profit"])
    finc_data = finc_data[finc_data['Code'].str[-2:] != 'BJ'].copy()
    all_data = prepare_latest_3_years_finc_data(finc_data)
    all_data = all_data[(all_data["AnnDate"] >= calc_sd) & (all_data["revenue"] >= SALES_EPSILON)].copy()
    all_data["earnings_var"] = all_data[["net_profit", "net_profit_last_year", "net_profit_last_2_year"]].std(axis=1, skipna=False) / \
                               all_data[["net_profit", "net_profit_last_year", "net_profit_last_2_year"]].abs().mean(axis=1, skipna=False)
    all_data["sales_var"] = all_data[["revenue", "revenue_last_year", "revenue_last_2_year"]].std(axis=1, skipna=False) / \
                               all_data[["revenue", "revenue_last_year", "revenue_last_2_year"]].mean(axis=1, skipna=False).abs()
    all_data.rename(columns={"AnnDate": "CalcDate"}, errors="raise", inplace=True)
    #
    rtn = _fwd_fill_finc_features(all_data[["CalcDate", "Code", "report_period", "earnings_var", "sales_var"]], ecd)
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both")].copy()
    return rtn


def calc_cash_flows_var(scd, ecd):
    FWD_FILL_WINSIZE = 500
    calc_sd = CALENDAR_UTIL.get_latest_n_dates(scd, FWD_FILL_WINSIZE + 1)[0]
    data_sd = CALENDAR_UTIL.get_n_years_before([calc_sd], 2)[0]
    finc_data = load_cashflow_by_wind(data_sd, ecd, ["op_net_cash"])
    finc_data = finc_data[finc_data['Code'].str[-2:] != 'BJ'].copy()
    #
    all_data = prepare_latest_3_years_finc_data(finc_data)
    all_data = all_data[all_data["AnnDate"] >= calc_sd].dropna(subset=["op_net_cash"])
    #
    all_data["cash_flows_var"] = all_data[["op_net_cash", "op_net_cash_last_year", "op_net_cash_last_2_year"]].std(axis=1, skipna=False) / \
                               all_data[["op_net_cash", "op_net_cash_last_year", "op_net_cash_last_2_year"]].abs().mean(axis=1, skipna=False)
    all_data.rename(columns={"AnnDate": "CalcDate"}, errors="raise", inplace=True)
    #
    rtn = _fwd_fill_finc_features(all_data[["CalcDate", "Code", "report_period", "cash_flows_var"]], ecd)
    #
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both")].copy()
    return rtn


def calc_earning_var(scd, ecd):
    earning_sales_var = calc_earnings_sales_var(scd, ecd)
    cash_var = calc_cash_flows_var(scd, ecd)
    rtn = pd.merge(earning_sales_var, cash_var, how='inner', on=['CalcDate', 'Code'])
    rtn = rtn.set_index(['CalcDate', 'Code'])
    rtn = apply_qtile_shrink(np.sqrt(rtn.dropna()))
    rtn = apply_zscore(rtn)
    rtn = rtn.sum(axis=1, skipna=False)
    rtn = apply_zscore(rtn).rename('Earnvar')
    return rtn