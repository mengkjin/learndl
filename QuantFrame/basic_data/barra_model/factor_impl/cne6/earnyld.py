from daily_bar.api import load_daily_bar_data
from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd
from .util import get_latest_year_rptprd_range, prepare_latest_year_finc_data
from basic_src_data.wind_tools.finance import load_income_by_wind, load_cashflow_by_wind
from .cne6_utils import apply_qtile_shrink, apply_zscore


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


def _calc_ma_cap(mv_data, ma_win_size):
    mv_data = mv_data[mv_data["total_value"] >= CAP_EPSILON].copy()
    rtn = mv_data.set_index(["CalcDate", "Code"])["total_value"].unstack().rolling(ma_win_size, min_periods=1).mean()
    #
    rtn = rtn.stack().rename("cap").reset_index(drop=False)
    return rtn


def calc_cash_earnings_to_price(root_path, scd, ecd):
    CAP_MA_WIN_SIZE = 30
    mv_sd = CALENDAR_UTIL.get_last_dates([scd], n=CAP_MA_WIN_SIZE)[0]
    total_value = _get_valuation_data(root_path, mv_sd, ecd, ["total_value"])
    cap_data = _calc_ma_cap(total_value, CAP_MA_WIN_SIZE)
    #
    FWD_FILL_WINSIZE = 500
    calc_sd = CALENDAR_UTIL.get_latest_n_dates(scd, FWD_FILL_WINSIZE + 1)[0]
    cum_stm = load_cashflow_by_wind(calc_sd, ecd, ["op_net_cash"])
    cum_stm = cum_stm[cum_stm['Code'].str[-2:] != 'BJ'].copy()
    finc_data = prepare_latest_year_finc_data(cum_stm)
    #
    cap_data["CalcDate_alias"] = cap_data["CalcDate"].str.replace("-", "").astype(int)
    finc_data["AnnDate_alias"] = finc_data["AnnDate"].str.replace("-", "").astype(int)
    rtn = pd.merge_asof(cap_data, finc_data[["AnnDate_alias", "Code", "report_period", "op_net_cash"]],
                        left_on='CalcDate_alias', right_on='AnnDate_alias', by=["Code"],
                        direction="backward", allow_exact_matches=True)
    date_df = rtn[["CalcDate"]].drop_duplicates()
    rtn = pd.merge(rtn, date_df.assign(min_rpt_prd=date_df["CalcDate"].apply(lambda x: get_latest_year_rptprd_range(x)[0]).astype(str)),
                   how="left", on=["CalcDate"]).query("report_period >= min_rpt_prd").\
        dropna(subset=["op_net_cash"], how="any")
    #
    rtn["cetop"] = rtn["op_net_cash"] / rtn["cap"]
    #
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both"), ["CalcDate", "Code", "cetop"]].copy()
    return rtn


def calc_earnings_to_price(root_path, scd, ecd):
    CAP_MA_WIN_SIZE = 30
    mv_sd = CALENDAR_UTIL.get_last_dates([scd], n=CAP_MA_WIN_SIZE)[0]
    total_value = _get_valuation_data(root_path, mv_sd, ecd, ["total_value"])
    cap_data = _calc_ma_cap(total_value, CAP_MA_WIN_SIZE)
    #
    FWD_FILL_WINSIZE = 500
    calc_sd = CALENDAR_UTIL.get_latest_n_dates(scd, FWD_FILL_WINSIZE + 1)[0]
    cum_stm = load_income_by_wind(calc_sd, ecd, ["net_profit"])
    cum_stm = cum_stm[cum_stm['Code'].str[-2:] != 'BJ'].copy()
    finc_data = prepare_latest_year_finc_data(cum_stm)
    #
    cap_data["CalcDate_alias"] = cap_data["CalcDate"].str.replace("-", "").astype(int)
    finc_data["AnnDate_alias"] = finc_data["AnnDate"].str.replace("-", "").astype(int)
    rtn = pd.merge_asof(cap_data, finc_data[["AnnDate_alias", "Code", "report_period", "net_profit"]],
                        left_on='CalcDate_alias', right_on='AnnDate_alias', by=["Code"],
                        direction="backward", allow_exact_matches=True)
    date_df = rtn[["CalcDate"]].drop_duplicates()
    rtn = pd.merge(rtn, date_df.assign(min_rpt_prd=date_df["CalcDate"].apply(lambda x: get_latest_year_rptprd_range(x)[0]).astype(str)),
                   how="left", on=["CalcDate"]).query("report_period >= min_rpt_prd").dropna(subset=["net_profit"])
    #
    rtn["etop"] = rtn["net_profit"] / rtn["cap"]
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both"), ["CalcDate", "Code", "etop"]].copy()
    return rtn


def calc_earnyld(root_path, scd, ecd):
    cash_earning_to_price = calc_cash_earnings_to_price(root_path, scd, ecd)
    profit_earning_to_price = calc_earnings_to_price(root_path, scd, ecd)
    earning_to_price = pd.merge(cash_earning_to_price, profit_earning_to_price, how='inner', on=['CalcDate', 'Code'])
    rtn = earning_to_price.set_index(['CalcDate', 'Code'])
    rtn = apply_qtile_shrink(rtn.dropna(how='any'))
    rtn = apply_zscore(rtn).sum(axis=1, skipna=False)
    rtn = apply_zscore(rtn).rename('Earnyld')
    return rtn