from financial_tools.report_periods import get_valid_nes_ann_rptprd_range
import pandas as pd
from basic_src_data.wind_tools.divnsplit import load_dividend_info
from basic_src_data.wind_tools.finance import load_income_by_wind
from events_system.calendar_util import CALENDAR_UTIL


NOT_PASS = '4'
NOT_IMPL = '12'
WIND_NOT_IMPL_CODE = [NOT_PASS, NOT_IMPL]


def _process_div_data(div_data):
    div_data = div_data[div_data["report_period"].str[4:].isin(["0331", "0630", "0930", "1231"])].copy()
    div_data = div_data[~(div_data["preplan_date"].str[:4].astype(int) >
                        (div_data["report_period"].str[:4].astype(int) + 1))].copy()
    div_data = div_data[~(
            (div_data["latest_ann_date"] == div_data["preplan_date"]) & div_data["progress"].isin(WIND_NOT_IMPL_CODE)
    )].copy()
    rtn = div_data.groupby(by=["preplan_date", "Code", "report_period"], as_index=False).agg(
        {"div_amount": "sum", "progress": "first", "latest_ann_date": "first"})
    return rtn


def _get_anndate_of_fst_year_rpt(scd, ecd):
    fin_data = load_income_by_wind(scd, ecd, ["net_profit"])[["AnnDate", "Code", "report_period"]]
    fin_data = fin_data[fin_data["report_period"].astype(int) <=
                        fin_data["AnnDate"].apply(lambda x: get_valid_nes_ann_rptprd_range(x)[1])].copy()
    year_fin = fin_data.loc[fin_data["report_period"].str[4:] == "1231"].\
        sort_values(["AnnDate", "Code", "report_period"]).drop_duplicates(["AnnDate", "Code"], keep="last")
    year_fin["report_period_alias"] = year_fin["report_period"].astype(int)
    year_fin = year_fin[year_fin["report_period_alias"] == year_fin.groupby(["Code"])["report_period_alias"].cummax()].\
        drop(columns=["report_period_alias"])
    statement_line = year_fin[["AnnDate", "Code", "report_period"]].sort_values(["AnnDate", "Code", "report_period"])\
        .drop_duplicates(["Code", "report_period"], keep="first")
    statement_line = statement_line[statement_line["report_period"].astype(int) >=
                                    statement_line["AnnDate"].apply(lambda x: get_valid_nes_ann_rptprd_range(x)[0])].copy()
    return statement_line


def _fill_reset_div(div_data, statement_line):
    statement_line = statement_line.copy()
    statement_line["AnnDate_alias"] = statement_line["AnnDate"].str.replace("-", "").astype(int)
    #
    preplan_data = div_data[["preplan_date", "Code", "report_period", "div_amount"]].copy()
    preplan_data["preplan_date_alias"] = preplan_data["preplan_date"].str.replace("-", "").astype(int)
    statement_data = pd.merge_asof(statement_line, preplan_data, left_on="AnnDate_alias", right_on="preplan_date_alias",
                                   by=["Code", "report_period"], direction="backward", allow_exact_matches=True)
    stm_with_no_div_pub = statement_data[statement_data["preplan_date"].isna()].copy()
    stm_with_no_div_pub["preplan_date"] = stm_with_no_div_pub["AnnDate"]
    stm_with_no_div_pub["div_amount"] = 0.0
    stm_with_no_div_pub['CalcDate'] = stm_with_no_div_pub['preplan_date']
    #
    preplan_data["CalcDate"] = preplan_data["preplan_date"]
    #
    not_impl_div = div_data[div_data["progress"].isin(WIND_NOT_IMPL_CODE)].copy()
    not_impl_div["CalcDate"] = not_impl_div["latest_ann_date"]
    not_impl_div["div_amount"] = 0.0
    #
    all_data = pd.concat((
        preplan_data[["CalcDate", "Code", "report_period", "div_amount", "preplan_date"]],
        stm_with_no_div_pub[['CalcDate', 'Code', 'report_period', 'div_amount', 'preplan_date']],
        not_impl_div[["CalcDate", "Code", "report_period", "div_amount", "preplan_date"]]),
        axis=0).sort_values(["CalcDate", "Code", "report_period", "preplan_date"])
    assert not all_data[["CalcDate", "Code", "report_period", 'preplan_date']].duplicated().any()
    return all_data


def _calc_div_by_code_for_dupl_data(df):
    rtn = []
    calc_date_list = df.loc[df["rpt_md"] == "1231", "CalcDate"].unique()
    assert len(calc_date_list) > 0
    for calc_date in calc_date_list:
        cum_row = df[df["CalcDate"] <= calc_date].copy()
        cum_row = cum_row.sort_values(["report_period", "preplan_date", "CalcDate"], ascending=True).\
            drop_duplicates(subset=["report_period", "preplan_date"], keep="last")
        rtn.append(
            [calc_date, cum_row["div_amount"].sum()]
        )
    rtn = pd.DataFrame(rtn, columns=["CalcDate", "div_amount"])
    return rtn


def _agg_year_div_impl(div_data):
    div_data = div_data.copy()
    div_data["rpt_year"] = div_data["report_period"].str[:4].astype(int)
    div_data['rpt_md'] = div_data['report_period'].str[4:]
    div_data = pd.merge(
        div_data, div_data.groupby(['Code', 'rpt_year'])['rpt_md'].max().rename('max_rpt_md'),
        how='left',
        on=['Code', 'rpt_year']
    )
    div_data = div_data[div_data['max_rpt_md'] == '1231'].drop(columns=['max_rpt_md'])
    #
    max_rptprd_for_code_rpt_year = div_data.groupby(['Code', 'rpt_year', 'report_period']).size().groupby(['Code', 'rpt_year']).max()
    dupl_code_year = max_rptprd_for_code_rpt_year[max_rptprd_for_code_rpt_year > 1].index.to_frame(index=False)
    uniq_code_year = max_rptprd_for_code_rpt_year[max_rptprd_for_code_rpt_year == 1].index.to_frame(index=False)
    dupl_data = pd.merge(div_data, dupl_code_year, how='inner', on=['Code', 'rpt_year']).sort_values(['CalcDate', 'Code', 'report_period', 'preplan_date'])
    unique_data = pd.merge(div_data, uniq_code_year, how='inner', on=['Code', 'rpt_year']).sort_values(['CalcDate', 'Code', 'report_period', 'preplan_date'])
    #
    dupl_div = dupl_data.groupby(["Code", "rpt_year"]).apply(_calc_div_by_code_for_dupl_data).reset_index(drop=False)
    #
    assert unique_data.groupby(['Code', 'rpt_year'])['rpt_md'].apply(lambda x: x.is_monotonic_increasing).all()
    unique_div = unique_data.groupby(["Code", "rpt_year"])[["CalcDate", "div_amount"]].agg(
        {"CalcDate": "max", "div_amount": "sum"}).reset_index(drop=False)
    #
    rtn = pd.concat((dupl_div, unique_div), axis=0)
    return rtn


def agg_year_div(scd, ecd):
    if scd[5:] < '08-31':
        div_start_date = str(int(scd[:4]) - 1) + '-04-01'
    else:
        div_start_date = scd[:4] + '-04-01'
    div_data = load_dividend_info(div_start_date, ecd)
    div_data = _process_div_data(div_data)
    statement_line = _get_anndate_of_fst_year_rpt(scd, ecd)
    div_data = _fill_reset_div(div_data, statement_line)
    rtn = _agg_year_div_impl(div_data)
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both"), ["CalcDate", "Code", "rpt_year", "div_amount"]].\
        sort_values(["CalcDate", "Code"])
    assert not rtn.duplicated(['CalcDate', 'Code']).any()
    return rtn