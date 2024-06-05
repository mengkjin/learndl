import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from .util import get_latest_year_rptprd_range, prepare_latest_year_finc_data
from basic_src_data.wind_tools.finance import load_balance_by_wind
from daily_bar.api import load_daily_bar_data
from .to_get_insurance_stock import get_insurance_stock
import numpy as np
from industry.api import load_industry_data
from .cne6_utils import apply_qtile_shrink, apply_zscore


ASSET_EPSILON = 1e7
EQUITY_EPSILON = 1e7
CAP_EPSILON = 1e8


def _get_valuation_data(root_path, scd, ecd, fld_list):
    total_value = load_daily_bar_data(root_path, "valuation",
                                      CALENDAR_UTIL.get_last_trading_dates([scd], inc_self_if_is_trdday=False)[0], ecd)
    total_value = total_value.rename(columns={"CalcDate": "trade_CalcDate"}, errors="raise")
    date_df = total_value[["trade_CalcDate"]].drop_duplicates()
    date_df["calendar_dates"] = date_df["trade_CalcDate"].apply(
        lambda x: CALENDAR_UTIL.get_ranged_dates(
            x, CALENDAR_UTIL.get_next_trading_dates([x], inc_self_if_is_trdday=False)[0])[:-1])
    rtn = pd.merge(total_value, date_df, on=["trade_CalcDate"], how="left").rename(
        columns={"calendar_dates": "CalcDate"}, errors="raise")
    rtn = rtn.explode("CalcDate").drop(columns=["trade_CalcDate"]).\
        sort_values(["CalcDate", "Code"])
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both"), ["CalcDate", "Code"] + fld_list].copy()
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


def _calc_ma_cap(mv_data, ma_win_size):
    mv_data = mv_data[mv_data["total_value"] >= CAP_EPSILON].copy()
    rtn = mv_data.set_index(["CalcDate", "Code"])["total_value"].unstack().rolling(ma_win_size, min_periods=1).mean()
    #
    rtn = rtn.stack().rename("cap").reset_index(drop=False)
    return rtn


def _calc_market_leverage(cap_data, finc_data):
    cap_data = cap_data.copy()
    finc_data = finc_data.copy()
    cap_data["CalcDate_alias"] = cap_data["CalcDate"].str.replace("-", "").astype(int)
    finc_data["AnnDate_alias"] = finc_data["AnnDate"].str.replace("-", "").astype(int)
    rtn = pd.merge_asof(cap_data, finc_data[["AnnDate_alias", "Code", "report_period", "non_cur_liab"]],
                        left_on='CalcDate_alias', right_on='AnnDate_alias', by=["Code"],
                        direction="backward", allow_exact_matches=True)
    date_df = rtn[["CalcDate"]].drop_duplicates()
    rtn = pd.merge(rtn, date_df.assign(
        min_rpt_prd=date_df["CalcDate"].apply(lambda x: get_latest_year_rptprd_range(x)[0]).astype(str)),
                   how="left", on=["CalcDate"]).query("report_period >= min_rpt_prd").dropna(subset=["non_cur_liab"])
    rtn["market_leverage"] = (rtn["non_cur_liab"]) / rtn["cap"]
    rtn = rtn[["CalcDate", "Code", "market_leverage"]].copy()
    return rtn


def _patch_bank_insurance_data(root_path, factor_data):
    assert factor_data["CalcDate"].is_monotonic_increasing
    scd, ecd = factor_data["CalcDate"].iloc[0], factor_data["CalcDate"].iloc[-1]
    industry_data = load_industry_data(root_path, scd, ecd, "citics_1", False)
    bank_stk = industry_data.query("citics_1 == 'Bank'").drop(columns=["citics_1"])
    insurance_stk = get_insurance_stock(scd, ecd)
    #
    bank_insurance = pd.concat((bank_stk, insurance_stk), axis=0).sort_values(["CalcDate", "Code"]).reset_index(drop=True)
    bank_insurance["is_bank_or_insurance"] = 1
    #
    rtn = pd.merge(factor_data, bank_insurance, how="left", on=["CalcDate", "Code"])
    rtn["is_bank_or_insurance"].fillna(0.0, inplace=True)
    rtn.loc[rtn["is_bank_or_insurance"] > 0, ["book_leverage", "market_leverage"]] = np.nan
    return rtn


def calc_leverage(root_path, scd, ecd):
    FWD_FILL_WINSIZE = 500
    calc_sd = CALENDAR_UTIL.get_latest_n_dates(scd, FWD_FILL_WINSIZE + 1)[0]
    balance = load_balance_by_wind(calc_sd, ecd, ["asset", "liability", "equity", "non_cur_liab"])
    balance = balance[balance['Code'].str[-2:] != 'BJ'].copy()
    finc_data = prepare_latest_year_finc_data(balance)
    #
    finc_data["debt_to_equity"] = finc_data["liability"] / finc_data["equity"].mask(finc_data["equity"] < EQUITY_EPSILON, other=np.nan)
    finc_data["book_leverage"] = (finc_data["non_cur_liab"]) / \
                                 finc_data["equity"].mask(finc_data["equity"] < EQUITY_EPSILON, other=np.nan)
    finc_data["CalcDate"] = finc_data["AnnDate"]
    #
    finc_leverage = _fwd_fill_finc_features(finc_data[["CalcDate", "Code", "report_period", "debt_to_equity",
                                                       "book_leverage"]].copy(), ecd)
    finc_leverage = finc_leverage[finc_leverage["CalcDate"].between(scd, ecd, inclusive="both")].copy()
    rtn = _patch_bank_insurance_data(root_path, finc_leverage)
    rtn = rtn[["CalcDate", "Code", "debt_to_equity", "book_leverage", 'is_bank_or_insurance']].set_index(['CalcDate', 'Code'])
    nonbank_rtn = rtn[rtn['is_bank_or_insurance'] < 0.5].drop(columns=['is_bank_or_insurance']).dropna(how='any')
    bi_rtn = rtn[rtn['is_bank_or_insurance'] > 0.5].drop(columns=['is_bank_or_insurance'])
    nonbank_rtn = apply_qtile_shrink(np.log(nonbank_rtn + 1))
    nonbank_rtn = apply_zscore(apply_zscore(nonbank_rtn).sum(axis=1, skipna=False))
    bi_rtn = apply_qtile_shrink(bi_rtn[['debt_to_equity']].copy())
    bi_rtn = apply_zscore(bi_rtn).iloc[:, 0]
    rtn = pd.concat((nonbank_rtn, bi_rtn)).sort_index().rename('Leverage')
    return rtn