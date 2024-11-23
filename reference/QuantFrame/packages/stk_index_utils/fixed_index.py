from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd
import numpy as np


def load_fixed_index_weight_data(root_path, index_nm, scd, ecd):
    if index_nm == "zero_rate":
        rtn = pd.DataFrame(columns=["CalcDate", "Code", "member_weight"])
    else:
        assert False
    return rtn


def load_fixed_index_level(root_path, index_nm, scd, ecd):
    if index_nm == "zero_rate":
        date_list = CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
        index_level = pd.DataFrame(zip(date_list, [1.0] * len(date_list)), columns=["CalcDate", "close_level"])
    else:
        assert False
    return index_level


def calc_fixed_index_ret_bias(root_path, index_nm, scd, ecd):
    calc_date_list = CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
    rtn = pd.Series(np.zeros(len(calc_date_list)), index=pd.Index(calc_date_list, name="CalcDate")).\
        rename("index_ret_bias")
    return rtn