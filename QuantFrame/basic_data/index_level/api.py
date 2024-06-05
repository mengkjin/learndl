from events_system.calendar_util import CALENDAR_UTIL
from .configs import get_data_path
from crosec_mem.data_vendor import DataVendor, get_data_info as _get_data_info


DATA_VENDOR = dict()


def load_index_level_data(root_path, scd, ecd, category, index_nms):
    key = category + '@' + root_path
    if key not in DATA_VENDOR:
        DATA_VENDOR[key] = DataVendor(get_data_path(root_path, category), "index_level")
    rtn = DATA_VENDOR[key].load_data(scd, ecd, expected_cal_type='trade')
    rtn = rtn[rtn["index_code"].isin(index_nms)].copy()
    loaded_dates = rtn['CalcDate'].unique().tolist()
    target_dates = CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
    assert loaded_dates == target_dates, \
        "  status::index_level>>api>>load data's dates from {0} to {1} are not enough for category {2}.".format(
            scd, ecd, category)
    return rtn


def get_data_info(root_path, category):
    data_path = get_data_path(root_path, category)
    rtn = _get_data_info(data_path)
    return rtn