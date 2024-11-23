from events_system.calendar_util import CALENDAR_UTIL
from stk_basic_info.api import load_stk_basic_info_data


def get_early_delist(root_path, scd, ecd):
    df = load_stk_basic_info_data(root_path, "description")
    df = df.loc[(df["delist_date"] >= scd) & (df["early_delist_date"] <= ecd), ["Code", "early_delist_date", "delist_date"]].copy()
    df['Date'] = df.apply(lambda x: CALENDAR_UTIL.get_ranged_dates(x['early_delist_date'], x['delist_date']), axis=1)
    rtn = df[['Code', 'Date']].explode('Date')
    rtn = rtn.loc[rtn['Date'].between(scd, ecd), ['Date', 'Code']].sort_values(['Date', 'Code'])
    rtn['remove_by_delist'] = 1
    return rtn