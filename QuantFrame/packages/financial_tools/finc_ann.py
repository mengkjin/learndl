import pandas as pd
from basic_src_data.wind_tools.finance import load_income_by_wind, load_express_data, load_notice_data


def _get_base_line_fin_data(fin_data):
    def _get_prd_rpd_range_for_nes(date_):
        y, last_y = date_[:4], str(int(date_[:4]) - 1)
        m = int(date_[5:7])
        if m == 6 or m == 7 or m == 8:
            rtn_ = y + '0630', y + '0630'
        elif m == 9 or m == 10:
            rtn_ = y + '0930', y + '0930'
        elif m == 12:
            rtn_ = y + '1231', y + '1231'
        elif m == 1 or m == 2:
            rtn_ = last_y + '1231', last_y + '1231'
        elif m == 3 or m == 4:
            rtn_ = last_y + '1231', y + '0331'
        else:
            assert False
        return int(rtn_[0]), int(rtn_[1])

    fin_data = fin_data[~fin_data["AnnDate"].str[5:7].astype(int).isin([5, 11])].copy()
    min_period = fin_data["AnnDate"].apply(lambda x: _get_prd_rpd_range_for_nes(x)[0]).astype(str)
    max_period = fin_data["AnnDate"].apply(lambda x: _get_prd_rpd_range_for_nes(x)[1]).astype(str)
    rtn = fin_data[(fin_data["report_period"] <= max_period) &
                   (fin_data["report_period"] >= min_period)
                   ].sort_values(["AnnDate", "Code", "report_period"], ascending=True)
    rtn["report_period"] = rtn["report_period"].astype(int)
    rtn = rtn[rtn["report_period"] == rtn.groupby(["Code"])["report_period"].cummax()].copy()
    rtn["report_period"] = rtn["report_period"].astype(str)
    return rtn


def get_valid_finc_ann(sad, ead):
    income_data = load_income_by_wind(sad, ead, ["net_profit"])
    express_data = load_express_data(sad, ead, ["net_profit"])
    notice_data = load_notice_data(sad, ead)
    notice_data.dropna(subset=['rate_max', 'rate_min', 'amt_min', 'amt_max'], how='all', inplace=True)
    all_fin_ann = pd.concat((
        notice_data[['AnnDate', 'Code', 'report_period']].drop_duplicates(),
        express_data[['AnnDate', 'Code', 'report_period']].drop_duplicates(),
        income_data[['AnnDate', 'Code', 'report_period']].drop_duplicates()
    ), axis=0)
    rtn = _get_base_line_fin_data(
        all_fin_ann.sort_values(['AnnDate', 'Code', 'report_period']).
            drop_duplicates(['AnnDate', 'Code', 'report_period']))
    return rtn