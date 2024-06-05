from daily_bar.api import load_daily_bar_data
import numpy as np
from financial_tools.report_periods import get_valid_nes_ann_rptprd_range
import pandas as pd
from basic_src_data.wind_tools.finance import load_balance_by_wind
from .year_div_agg import agg_year_div
from events_system.calendar_util import CALENDAR_UTIL
from .cne6_utils import apply_zscore, apply_qtile_side_by_side_shrink


EQUITY_EPSILON = 1e8
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


def _get_latest_year_fin_data(start_date, ecd):
    fin_data = load_balance_by_wind(start_date, ecd, ['equity'])
    fin_data = fin_data[fin_data["report_period"].astype(int) <=
                        fin_data["AnnDate"].apply(lambda x: get_valid_nes_ann_rptprd_range(x)[1])].copy()
    year_fin_data = fin_data.loc[fin_data["report_period"].str[4:] == "1231",
                           ["AnnDate", "Code", "report_period", "equity"]].\
        sort_values(["AnnDate", "Code", "report_period"]).drop_duplicates(["AnnDate", "Code"], keep="last")
    year_fin_data["report_period_alias"] = year_fin_data["report_period"].astype(int)
    year_fin_data = year_fin_data[year_fin_data["report_period_alias"] == year_fin_data.groupby(["Code"])["report_period_alias"].cummax()].\
        drop(columns=["report_period_alias"])
    rtn = year_fin_data[year_fin_data['report_period'].str[:4].astype(int) + 2 > year_fin_data['AnnDate'].str[:4].astype(int)].copy()
    return rtn


def _merge_div_with_eq(year_div, year_fin_data):
    year_fin_data = year_fin_data.copy()
    year_fin_data["rpt_year"] = year_fin_data["report_period"].str[:4].astype(year_div.dtypes['rpt_year'])
    div_eq = pd.merge(
        year_div.rename(columns={'CalcDate': 'div_calc_date'}),
        year_fin_data[['AnnDate', 'Code', 'rpt_year', 'equity']],
        how='left', on=['Code', 'rpt_year'])
    div_eq['CalcDate'] = div_eq['div_calc_date']
    notna_ann_flg = div_eq['AnnDate'].notnull()
    div_eq.loc[notna_ann_flg, 'CalcDate'] = div_eq.loc[notna_ann_flg, ['div_calc_date', 'AnnDate']].max(axis=1)
    div_eq = div_eq.sort_values(['Code', 'rpt_year', 'CalcDate', 'AnnDate']).drop_duplicates(['Code', 'rpt_year', 'CalcDate'], keep='last')
    rtn = div_eq.sort_values(['CalcDate', 'Code', 'rpt_year'])
    assert not rtn.duplicated(['CalcDate', 'Code']).any()
    return rtn


def _fwd_fill_feature(div_df, ecd):
    date_range = CALENDAR_UTIL.get_ranged_dates(div_df["CalcDate"].min(), ecd)
    fwd_col = div_df.columns.drop(["CalcDate", "Code"])
    daily_div = div_df.set_index(["CalcDate", "Code"])[fwd_col].unstack().\
        reindex(date_range).fillna(method="ffill").stack().reset_index(drop=False)
    daily_div["rpt_year"] = daily_div["rpt_year"].astype(int)
    daily_div = daily_div[daily_div["CalcDate"] <= (daily_div["rpt_year"] + 2).astype(str) + "-04-30"].\
        drop(columns=["rpt_year"])
    return daily_div


def calc_dtop(root_path, scd, ecd):
    CAP_MA_WIN_SIZE = 30
    mv_sd = CALENDAR_UTIL.get_last_dates([scd], n=CAP_MA_WIN_SIZE)[0]
    valuation = _get_valuation_data(root_path, mv_sd, ecd, ["total_value", "book_value"])
    valuation = valuation[valuation["Code"].str[-2:] != 'BJ'].copy()
    cap_data = _calc_ma_cap(valuation[["CalcDate", "Code", "total_value"]], CAP_MA_WIN_SIZE)
    valuation = pd.merge(cap_data, valuation[["CalcDate", "Code", "book_value"]], how="left", on=["CalcDate", "Code"])
    #
    if scd[5:] <= '04-30':
        div_sd = str(int(scd[:4]) - 1) + '-01-01'
    else:
        div_sd = str(int(scd[:4])) + '-01-01'
    year_div = agg_year_div(div_sd, ecd)
    assert (year_div['div_amount'] >= 0.0).all()
    year_fin_data = _get_latest_year_fin_data(div_sd, ecd)
    div_rate_data = _merge_div_with_eq(year_div, year_fin_data)
    #
    daily_div = _fwd_fill_feature(div_rate_data, ecd)
    daily_div = daily_div[daily_div['CalcDate'].between(scd, ecd)].copy()
    #
    rtn = pd.merge(valuation, daily_div, how="inner", on=["CalcDate", "Code"])
    rtn["dtop"] = rtn["div_amount"] / rtn["cap"]
    valid_equity_flg = (rtn["equity"] > EQUITY_EPSILON) & (rtn["book_value"] > EQUITY_EPSILON)
    rtn.loc[valid_equity_flg, "dtop"] = rtn.loc[valid_equity_flg, "dtop"] * rtn.loc[valid_equity_flg, "book_value"] / \
                                        rtn.loc[valid_equity_flg, "equity"]
    assert (rtn['dtop'] >= 0.0).all()
    rtn = rtn[['CalcDate', 'Code', 'dtop']].sort_values(['CalcDate', 'Code'])
    return rtn


def calc_divyld(root_path, scd, ecd):
    dtop = calc_dtop(root_path, scd, ecd).set_index(['CalcDate', 'Code']).dropna()
    dtop = np.sqrt(np.sqrt(dtop))
    rtn = apply_qtile_side_by_side_shrink(dtop)
    rtn = apply_zscore(rtn)['dtop'].rename('Divyld')
    return rtn