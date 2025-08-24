import numpy as np
from events_system.calendar_util import CALENDAR_UTIL
from crosec_mem.data_vendor import DataVendor, get_data_info as _get_data_info
from .configs import *


DATA_VENDOR = dict()


def query_index_weight_data(root_path, category, index_nms_, sys_id_list_, calc_date_):
    key = category + '@' + root_path
    if key not in DATA_VENDOR:
        DATA_VENDOR[key] = DataVendor(get_data_path(root_path, category), "index_weight")
    data_vendor = DATA_VENDOR[key]
    int_array, val_array = data_vendor.query_by_stk_sysid_on_calc_date(index_nms_, sys_id_list_, calc_date_)
    has_val_flg = int_array >= 0
    rtn = np.full((len(index_nms_), len(sys_id_list_)), 0.0, dtype=float)
    rtn[:, has_val_flg] = val_array[:, int_array[has_val_flg]]
    return rtn


def load_index_weight_data(root_path, category, index_nms, scd, ecd):
    key = category + '@' + root_path
    if key not in DATA_VENDOR:
        DATA_VENDOR[key] = DataVendor(get_data_path(root_path, category), "index_weight")
    rtn = DATA_VENDOR[key].load_data(scd, ecd, expected_cal_type='trade')
    rtn = rtn[["CalcDate", "Code"] + index_nms].copy()
    loaded_dates = rtn['CalcDate'].unique().tolist()
    target_dates = CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
    assert loaded_dates == target_dates, \
        "  status::index_weight>>api>>load data's dates from {0} to {1} are not enough for category {2}.".format(
            scd, ecd, category)
    return rtn


def get_data_info(root_path, category):
    data_path = get_data_path(root_path, category)
    rtn = _get_data_info(data_path)
    return rtn