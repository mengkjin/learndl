import pandas as pd


def prepare_latest_3_years_finc_data(finc_data):
    assert pd.Index(["AnnDate", "Code", "report_period"]).difference(finc_data.columns).empty
    finc_data = finc_data.sort_values(["AnnDate", "Code", "report_period"])
    year_finc = finc_data[finc_data["report_period"].str[4:] == '1231'].copy()
    #
    year_finc["rpt_prd_alias"] = year_finc["report_period"].astype(int)
    rtn = year_finc[year_finc["rpt_prd_alias"] == year_finc.groupby(
        ["Code"])["rpt_prd_alias"].cummax()].drop(columns=["rpt_prd_alias"])
    rtn = rtn.sort_values(["AnnDate", "Code", "report_period"]).drop_duplicates(["AnnDate", "Code"], keep="last")
    year_finc.drop(columns=["rpt_prd_alias"], inplace=True)
    #
    rtn["rpt_prd_last_year"] = (rtn["report_period"].str[:4].astype(int) - 1).astype(str) + "1231"
    rtn["AnnDate_alias"] = rtn["AnnDate"].str.replace("-", "").astype(int)
    year_finc["AnnDate_alias"] = year_finc["AnnDate"].str.replace("-", "").astype(int)
    rtn = pd.merge_asof(rtn, year_finc,
                        on="AnnDate_alias", left_by=["Code", "rpt_prd_last_year"], right_by=["Code", "report_period"],
                        suffixes=("", "_last_year"), direction="backward", allow_exact_matches=True)
    rtn["rpt_prd_last_2_year"] = (rtn["report_period"].str[:4].astype(int) - 2).astype(
        str) + "1231"
    rtn = pd.merge_asof(rtn, year_finc,
                        on="AnnDate_alias", left_by=["Code", "rpt_prd_last_2_year"], right_by=["Code", "report_period"],
                        suffixes=("", "_last_2_year"), direction="backward", allow_exact_matches=True)
    rtn.drop(columns=["AnnDate_alias"], inplace=True)
    return rtn


def prepare_latest_year_finc_data(finc_data):
    finc_data = finc_data.sort_values(["AnnDate", "Code", "report_period"])
    year_finc = finc_data[finc_data["report_period"].str[4:] == '1231'].copy()
    #
    year_finc["rpt_prd_alias"] = year_finc["report_period"].astype(int)
    rtn = year_finc[year_finc["rpt_prd_alias"] == year_finc.groupby(
        ["Code"])["rpt_prd_alias"].cummax()].drop(columns=["rpt_prd_alias"])
    rtn = rtn.sort_values(["AnnDate", "Code", "report_period"]).drop_duplicates(["AnnDate", "Code"], keep="last")
    return rtn


def calc_latest_ttm_finc_from_cum_data(inc_csh_finc_data):
    inc_csh_flds = inc_csh_finc_data.columns.drop(["AnnDate", "Code", "report_period"]).tolist()
    inc_csh_finc_data = inc_csh_finc_data.sort_values(["AnnDate", "Code", "report_period"])
    #
    inc_csh_finc_data["AnnDate_alias"] = inc_csh_finc_data["AnnDate"].str.replace("-", "").astype(int)
    inc_csh_finc_data["same_rpd_last_year"] = (inc_csh_finc_data["report_period"].str[:4].astype(int) - 1).astype(str) + \
                                              inc_csh_finc_data["report_period"].str[4:]
    ttm_finc = pd.merge_asof(inc_csh_finc_data,
                             inc_csh_finc_data[["AnnDate_alias", "Code", "report_period"] + inc_csh_flds],
                             on="AnnDate_alias",
                             left_by=["Code", "same_rpd_last_year"], right_by=["Code", "report_period"],
                             suffixes=("", "_same_rpd_last_year"), direction="backward", allow_exact_matches=True)
    ttm_finc["last_year_rpd"] = (ttm_finc["report_period"].str[:4].astype(int) - 1).astype(str) + "1231"
    ttm_finc = pd.merge_asof(ttm_finc,
                             inc_csh_finc_data[["AnnDate_alias", "Code", "report_period"] + inc_csh_flds],
                             on="AnnDate_alias", left_by=["Code", "last_year_rpd"], right_by=["Code", "report_period"],
                             suffixes=("", "_last_year"), direction="backward", allow_exact_matches=True)
    ttm_flds = ["ttm_" + nm for nm in inc_csh_flds]
    ttm_finc[ttm_flds] = ttm_finc[inc_csh_flds].values + ttm_finc[[nm + "_last_year" for nm in inc_csh_flds]].values - \
                                    ttm_finc[[nm + "_same_rpd_last_year" for nm in inc_csh_flds]].values
    ttm_finc = ttm_finc[["AnnDate", "Code", "report_period"] + ttm_flds].dropna(subset=ttm_flds, how="all")
    #
    ttm_finc["rpt_prd_alias"] = ttm_finc["report_period"].astype(int)
    rtn = ttm_finc[ttm_finc["rpt_prd_alias"] == ttm_finc.groupby(
        ["Code"])["rpt_prd_alias"].cummax()].drop(columns=["rpt_prd_alias"])
    rtn = rtn.sort_values(["AnnDate", "Code", "report_period"]).drop_duplicates(["AnnDate", "Code"], keep="last")
    return rtn


def get_latest_year_rptprd_range(date):
    y, last_y = str(int(date[:4]) - 1), str(int(date[:4]) - 2)
    m = int(date[5:7])
    if m == 1 or m == 2 or m == 3 or m == 4:
        rtn = last_y + '1231', y + '1231'
    elif m == 5 or m == 6 or m == 7 or m == 8 or m == 9 or m == 10 or m == 11 or m == 12:
        rtn = y + '1231', y + '1231'
    else:
        assert False
    return int(rtn[0]), int(rtn[1])


def get_statement_rptprd_range(date):
    y, last_y = date[:4], str(int(date[:4]) - 1)
    m = int(date[5:7])
    if m == 5 or m == 6:
        rtn = y + '0331', y + '0331'
    elif m == 7 or m == 8:
        rtn = y + '0331', y + '0630'
    elif m == 9:
        rtn = y + '0630', y + '0630'
    elif m == 10:
        rtn = y + '0630', y + '0930'
    elif m == 11 or m == 12:
        rtn = y + '0930', y + '0930'
    elif m == 1 or m == 2 or m == 3:
        rtn = last_y + '0930', last_y + '1231'
    elif m == 4:
        rtn = last_y + '0930', y + '0331'
    else:
        assert False
    return int(rtn[0]), int(rtn[1])