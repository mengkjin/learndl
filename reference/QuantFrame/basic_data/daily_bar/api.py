import numpy as np
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from .configs import *
from crosec_mem.data_vendor import DataVendor


DATA_VENDOR = dict()


def query_dbar_flds_data(root_path_, category_, fld_list_, sys_id_list_, calc_date_):
    key = category_ + '@' + root_path_
    if key not in DATA_VENDOR:
        DATA_VENDOR[key] = DataVendor(get_data_path(root_path_, category_), "daily_bar>>{0}".format(category_))
    data_vendor = DATA_VENDOR[key]
    assert isinstance(fld_list_, list) and len(fld_list_) == len(set(fld_list_))
    rtn = data_vendor.query_by_stk_sysid_on_calc_date(fld_list_, sys_id_list_, calc_date_)
    return rtn


def query_dbar_flds_data_w_dval(root_path_, category_, fld_list_, sys_id_list_, calc_date_, default_val_):
    key = category_ + '@' + root_path_
    if key not in DATA_VENDOR:
        DATA_VENDOR[key] = DataVendor(get_data_path(root_path_, category_), "daily_bar>>{0}".format(category_))
    data_vendor = DATA_VENDOR[key]
    assert isinstance(fld_list_, list) and len(fld_list_) == len(set(fld_list_))
    int_array, val_array = data_vendor.query_by_stk_sysid_on_calc_date(fld_list_, sys_id_list_, calc_date_)
    has_val_flg = int_array >= 0
    rtn = np.full((len(fld_list_), len(sys_id_list_)), default_val_, dtype=float)
    rtn[:, has_val_flg] = val_array[:, int_array[has_val_flg]]
    return rtn


def load_daily_bar_data(root_path, category, scd, ecd, calendar_type="trade"):
    assert scd >= '2003-01-01'
    assert calendar_type in ("trade", "calendar")
    key = category + '@' + root_path
    if key not in DATA_VENDOR:
        DATA_VENDOR[key] = DataVendor(get_data_path(root_path, category), "daily_bar>>{0}".format(category))
    #
    if calendar_type == "trade":
        target_date_list = CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
    else:
        target_date_list = CALENDAR_UTIL.get_ranged_dates(scd, ecd)
    #
    if category == "basic":
        assert calendar_type == "trade"
        rtn = DATA_VENDOR[key].load_data(scd, ecd, expected_cal_type="trade")
    elif category == "valuation":
        rtn = DATA_VENDOR[key].load_data(scd, ecd, expected_cal_type="full")
        if calendar_type == "trade":
            rtn = rtn[rtn["CalcDate"].isin(target_date_list)].copy()
    else:
        assert False
    assert rtn['CalcDate'].unique().tolist() == target_date_list, \
        "  status::daily_bar>>api>>load data's dates from {0} to {1} are not enough for category {2}.".format(
            scd, ecd, category)
    return rtn


def get_data_info(root_path, category):
    data_path = get_data_path(root_path, category)
    files = os.listdir(data_path)
    files = [f for f in files if os.path.splitext(f)[1] == '.h5']
    files.sort()
    first_file, last_file = files[0], files[-1]
    first_df = pd.read_hdf(os.path.join(data_path, first_file), key='df')
    first_calc_date = first_df['CalcDate'].iloc[0]
    last_df = pd.read_hdf(os.path.join(data_path, last_file), key='df')
    last_calc_date = last_df['CalcDate'].iloc[-1]
    rtn = {
        'first_calc_date': first_calc_date,
        'last_calc_date': last_calc_date
    }
    return rtn