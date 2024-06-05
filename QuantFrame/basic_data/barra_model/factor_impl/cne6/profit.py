import pandas as pd
import numpy as np
from events_system.calendar_util import CALENDAR_UTIL
from .util import get_latest_year_rptprd_range, prepare_latest_year_finc_data
from basic_src_data.wind_tools.finance import load_balance_by_wind, load_income_by_wind
from .cne6_utils import apply_zscore, apply_qtile_shrink


ASSET_EPSILON = 1e7
REVENUE_EPSILON = 1e6


def _load_finc_data(sad, ead):
    FIN_DATA_KEY = ["AnnDate", "Code", "report_period"]
    income = load_income_by_wind(sad, ead, ['net_profit', 'revenue', 'cost', 'op_cost', 'op_revenue'])
    income = income[income['Code'].str[-2:] != 'BJ'].copy()
    balance = load_balance_by_wind(sad, ead, ['asset'])
    balance = balance[balance['Code'].str[-2:] != 'BJ'].copy()
    statement = pd.merge(income[FIN_DATA_KEY], balance[FIN_DATA_KEY], on=FIN_DATA_KEY, how="outer").sort_values(FIN_DATA_KEY)
    #
    statement["AnnDate_alias"] = statement["AnnDate"].str.replace("-", "").astype(int)
    income["AnnDate_alias"] = income["AnnDate"].str.replace("-", "").astype(int)
    rtn = pd.merge_asof(statement, income.drop(columns=["AnnDate"]),
                        on=["AnnDate_alias"], by=["report_period", "Code"],
                        direction="backward", allow_exact_matches=True)
    balance["AnnDate_alias"] = balance["AnnDate"].str.replace("-", "").astype(int)
    rtn = pd.merge_asof(rtn, balance.drop(columns=["AnnDate"]),
                        on=["AnnDate_alias"], by=["report_period", "Code"],
                        direction="backward", allow_exact_matches=True)
    rtn = rtn.sort_values(FIN_DATA_KEY).reset_index(drop=True)
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


def calc_profit(scd, ecd):
    FWD_FILL_WINSIZE = 500
    data_sd = CALENDAR_UTIL.get_latest_n_dates(scd, FWD_FILL_WINSIZE + 1)[0]
    cum_stm = _load_finc_data(data_sd, ecd)
    all_data = prepare_latest_year_finc_data(cum_stm)
    #
    all_data["asset_turnover"] = all_data["revenue"] / all_data["asset"].mask(all_data["asset"] < ASSET_EPSILON, other=np.nan)
    all_data["roa"] = all_data["net_profit"] / all_data["asset"].mask(all_data["asset"] < ASSET_EPSILON, other=np.nan)
    all_data["gross_profit"] = (all_data["op_revenue"] - all_data["op_cost"]) / all_data["asset"].mask(
        all_data["asset"] < ASSET_EPSILON, other=np.nan)
    all_data.loc[all_data["gross_profit"].isna(), "gross_profit"] = \
        (all_data.loc[all_data["gross_profit"].isna(), "revenue"] - all_data.loc[all_data["gross_profit"].isna(), "cost"]) / \
        all_data.loc[all_data["gross_profit"].isna(), "asset"].mask(all_data.loc[all_data["gross_profit"].isna(), "asset"] < ASSET_EPSILON, other=np.nan)
    all_data["gross_profit_margin"] = (all_data["op_revenue"] - all_data["op_cost"]) / all_data["op_revenue"].mask(
        all_data["op_revenue"] < REVENUE_EPSILON, other=np.nan)
    all_data.loc[all_data["gross_profit_margin"].isna(), "gross_profit_margin"] = \
        (all_data.loc[all_data["gross_profit_margin"].isna(), "revenue"] - all_data.loc[all_data["gross_profit_margin"].isna(), "cost"]) / \
        all_data.loc[all_data["gross_profit_margin"].isna(), "revenue"].mask(all_data.loc[all_data["gross_profit_margin"].isna(), "revenue"] < REVENUE_EPSILON, other=np.nan)
    all_data.dropna(subset=["asset_turnover", "gross_profit", "gross_profit_margin", "roa"], how="all", inplace=True)
    #
    rtn = all_data[["AnnDate", "Code", "report_period", "asset_turnover", "roa",
                    "gross_profit", "gross_profit_margin"]].rename(columns={"AnnDate": "CalcDate"}, errors="raise")
    rtn = _fwd_fill_finc_features(rtn, ecd)
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both")].set_index(['CalcDate', 'Code']).dropna(how='any')
    rtn = apply_qtile_shrink(rtn)
    rtn = apply_zscore(rtn).mean(axis=1)
    rtn = apply_zscore(rtn).rename('Profit')
    return rtn