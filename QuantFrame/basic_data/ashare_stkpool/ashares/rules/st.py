from basic_src_data.wind_tools.basic import get_st_stocks_from_winddf
from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd


def get_st_filter(scd, ecd):
    lookback_winsize = 90
    data_sd = CALENDAR_UTIL.get_latest_n_dates(scd, lookback_winsize + 1)[0]
    st_stk_data = get_st_stocks_from_winddf(data_sd, ecd)
    st_stk_data["is_st"] = 1.0
    date_list = CALENDAR_UTIL.get_ranged_dates(data_sd, ecd)
    rtn = st_stk_data.set_index(["Date", "Code"])["is_st"].unstack().reindex(pd.Index(date_list, name="CalcDate")).\
              fillna(method="ffill", limit=lookback_winsize).loc[scd: ecd]
    rtn = rtn.stack().rename("is_st_within_90d").reset_index(drop=False)
    return rtn

