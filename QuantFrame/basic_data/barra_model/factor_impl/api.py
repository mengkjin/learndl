from events_system.calendar_util import CALENDAR_UTIL
from crosec_mem.data_vendor import DataVendor
from .configs import get_data_path
import pandas as pd


DATA_VENDOR = dict()


def load_barra_data(root_path, barra_type, scd, ecd, as_key=False):  # TODO: should change (scd_, ecd_) to date_list?
    assert scd >= '2003-01-01'
    key = root_path + '@' + barra_type
    if key not in DATA_VENDOR:
        DATA_VENDOR[key] = DataVendor(get_data_path(root_path, barra_type), 'barra')
    rtn = DATA_VENDOR[key].load_data(scd, ecd, expected_cal_type='full', store_type='ftr')
    loaded_dates = rtn['CalcDate'].unique().tolist()
    target_dates = CALENDAR_UTIL.get_ranged_dates(scd, ecd)
    assert loaded_dates == target_dates, \
        "  status::barra>>api>>load barra dates from {0} to {1} are not enough.".format(scd, ecd)
    if as_key:
        rtn.set_index(['CalcDate', 'Code'], inplace=True)
    return rtn


def load_barra_data_by_dates(root_path, date_list, barra_type='cne6'):
    assert date_list == sorted(date_list)
    scd, ecd = date_list[0], date_list[-1]
    barra_data = load_barra_data(root_path, barra_type, scd, ecd)
    rtn = barra_data[barra_data['CalcDate'].isin(date_list)].copy()
    return rtn


def merge_with_barra_data(root_path, df, barra_type, style_list, is_dropna=True):
    assert df['CalcDate'].is_monotonic_increasing
    barra_data = load_barra_data(root_path, barra_type, df['CalcDate'].iloc[0], df['CalcDate'].iloc[-1])
    if style_list is None:
        rtn = pd.merge(df, barra_data[['CalcDate', 'Code', 'INDUSTRY.citics_1']], how='left', on=['CalcDate', 'Code'])
    else:
        rtn = pd.merge(df, barra_data[['CalcDate', 'Code', 'INDUSTRY.citics_1'] + style_list], how='left', on=['CalcDate', 'Code'])
    if is_dropna:
        rtn.dropna(subset=['INDUSTRY.citics_1'], inplace=True)
    return rtn