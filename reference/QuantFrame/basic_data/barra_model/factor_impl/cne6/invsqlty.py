from .util import get_latest_year_rptprd_range, prepare_latest_3_years_finc_data
from basic_src_data.wind_tools.finance import load_balance_by_wind
from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd
import numpy as np
from .cne6_utils import apply_qtile_shrink, apply_zscore


ASSET_EPSILON = 1e7


def _calc_asset_grw(data):
    asset_data = data.set_index(["AnnDate", "Code", "report_period"])[[
        "asset", "asset_last_year", "asset_last_2_year"]].dropna(how="any")
    x = np.array([1, 2, 3])
    beta = asset_data.sub(asset_data.mean(axis=1), axis=0).dot(x - x.mean()) / (x - x.mean()).dot(x - x.mean())
    rtn = - beta / asset_data.mean(axis=1)
    rtn.rename("agro", inplace=True)
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


def calc_agro(scd, ecd):
    FWD_FILL_WINSIZE = 500
    calc_sd = CALENDAR_UTIL.get_latest_n_dates(scd, FWD_FILL_WINSIZE + 1)[0]
    data_sd = CALENDAR_UTIL.get_n_years_before([calc_sd], 2)[0]
    finc_data = load_balance_by_wind(data_sd, ecd, ["asset"])
    finc_data = finc_data[finc_data['Code'].str[-2:] != 'BJ'].copy()
    #
    all_data = prepare_latest_3_years_finc_data(finc_data)
    all_data = all_data[(all_data["asset"] > ASSET_EPSILON) & (all_data["AnnDate"] >= calc_sd)].copy()
    #
    agro = _calc_asset_grw(all_data)
    rtn = agro.reset_index(drop=False).rename(columns={"AnnDate": "CalcDate"}, errors="raise")
    rtn = _fwd_fill_finc_features(rtn, ecd)
    #
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both"), ["CalcDate", "Code", "agro"]].copy()
    return rtn


def calc_invsqlty(scd, ecd):
    agro = calc_agro(scd, ecd)
    agro = apply_qtile_shrink(agro.set_index(['CalcDate', 'Code']).dropna())
    rtn = apply_zscore(agro)['agro'].rename('Invsqlty')
    return rtn
