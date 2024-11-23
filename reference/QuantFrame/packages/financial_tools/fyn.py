import pandas as pd
from basic_src_data.wind_tools.finance import load_income_by_wind, load_express_data
from events_system.calendar_util import CALENDAR_UTIL


def get_fy1(start_date, end_date):
    look_back_start_date = str(int(start_date[:4])-1) + "-01-01"
    income_data = load_income_by_wind(look_back_start_date, end_date, ["net_profit"])
    express_data = load_express_data(look_back_start_date, end_date, ["net_profit"])
    fin_data = pd.concat((
        express_data[["AnnDate", "Code", "report_period"]],
        income_data[["AnnDate", "Code", "report_period"]]),
        axis=0)  # AnnDate, Code report_period
    #
    fin_data.sort_values(["AnnDate", "Code", "report_period"], ascending=True, inplace=True)
    fin_data.drop_duplicates(["AnnDate", "Code"], keep="last", inplace=True)
    fin_data["report_period"] = fin_data["report_period"].astype(int)
    fin_data.set_index(['AnnDate', 'Code'], inplace=True)
    fin_data["max_period"] = fin_data.groupby(level=['Code']).cummax()
    fin_data = fin_data[fin_data["report_period"] == fin_data["max_period"]].copy()
    fin_data["report_period"] = fin_data["report_period"].astype(str)
    #
    fin_data.reset_index(drop=False, inplace=True)
    fy_data = fin_data[['AnnDate', 'Code', 'report_period']].set_index(['AnnDate', 'Code']).unstack().reindex(
        CALENDAR_UTIL.get_ranged_dates(look_back_start_date, end_date)
    ).fillna(method='ffill').stack().reset_index(drop=False)
    fy_data['fy1'] = fy_data['AnnDate'].str[:4].astype(int)
    fy_data['fy0'] = fy_data['fy1'] - 1
    flg = (fy_data['AnnDate'] <= (fy_data['fy1'].astype(str) + '-04-30')) & (
            fy_data['report_period'].astype(str) < (fy_data['fy0'].astype(str) + '1231'))
    fy_data.loc[flg, 'fy1'] = fy_data.loc[flg, 'fy0']
    fy_data.rename(columns={"AnnDate": "CalcDate"}, inplace=True)
    fy_data["fy1"] = fy_data["fy1"].astype(int)
    rtn = fy_data.loc[fy_data["CalcDate"].between(start_date, end_date, inclusive="both"), ['CalcDate', 'Code', 'fy1']].copy()
    return rtn