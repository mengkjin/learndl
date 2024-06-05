from events_system.calendar_util import CALENDAR_UTIL
from industry.api import load_industry_data


def calc_industry(root_path, scd, ecd):
    rtn = load_industry_data(root_path, scd, ecd, 'citics_1', as_sys_id=False)
    rtn.rename(columns={'citics_1': 'industry'}, inplace=True)
    rtn.dropna(how='any', inplace=True)
    assert rtn['CalcDate'].drop_duplicates().tolist() == CALENDAR_UTIL.get_ranged_dates(scd, ecd)
    rtn = rtn[(rtn['Code'].str[0] != 'T') & (rtn['Code'].str[-2:] != 'BJ')].copy()
    return rtn