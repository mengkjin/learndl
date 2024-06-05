import pandas as pd
from .configs import *
from ashare_industry_id_sys.api import get_sys_id_from_en
from events_system.calendar_util import CALENDAR_UTIL
from crosec_mem.data_vendor import DataVendor
from industry_tools.sector import get_industry_sector_map


DATA_VENDOR = dict()


def load_industry_data(root_path, scd, ecd, industry_type, as_sys_id=False):
    assert scd >= START_CALC_DATE
    category, levels = industry_type.split("_")
    path = get_path(root_path, category)
    if path not in DATA_VENDOR:
        DATA_VENDOR[path] = DataVendor(path, "industry")
    rtn = DATA_VENDOR[path].load_data(scd, ecd, store_type="ftr")
    ind_names = [category + "_" + lv for lv in levels.split(",")]
    rtn = rtn[["CalcDate", "Code"] + ind_names].copy()
    loaded_dates = rtn['CalcDate'].unique().tolist()
    target_dates = CALENDAR_UTIL.get_ranged_dates(scd, ecd)
    assert loaded_dates == target_dates, \
        "  status::industry>>api>>load industry dates from {0} to {1} are not enough.".format(scd, ecd)
    if as_sys_id:
        industry_id = get_sys_id_from_en(industry_type, rtn[industry_type].tolist())
        rtn[industry_type] = industry_id
    return rtn


def load_industry_by_chinese(root_path, scd, ecd, industry_type):
    assert industry_type == "citics_1"
    ind_data = load_industry_data(root_path, scd, ecd, industry_type)
    sector_data = get_industry_sector_map("citics_1")
    ind_data = pd.merge(ind_data, sector_data, on=["citics_1"], how="left")
    ind_data = ind_data[["CalcDate", "Code", "citics_1_chn", "sector_name_chn"]].copy()
    ind_data.rename(columns={"citics_1_chn": "citics_1", "sector_name_chn": "sector_name"}, errors="raise", inplace=True)
    return ind_data


def select_industry(root_path, df, industry_type, ind_list, as_key=False):
    if as_key:
        df = df.reset_index()
    else:
        df = df.copy()
    assert df['CalcDate'].is_monotonic
    scd, ecd = df['CalcDate'].iloc[0], df['CalcDate'].iloc[-1]
    ind = load_industry_data(root_path, scd, ecd, industry_type)
    df = pd.merge(df, ind, how='left', on=['CalcDate', 'Code'])
    rtn = df[df[industry_type].isin(ind_list)].drop(columns=[industry_type])
    if as_key:
        rtn.set_index(['CalcDate', 'Code'], inplace=True)
    return rtn


def remove_industry(root_path, df, industry_type, ind_list, as_key=False):
    if as_key:
        df = df.reset_index()
    else:
        df = df.copy()
    assert df['CalcDate'].is_monotonic
    scd, ecd = df['CalcDate'].iloc[0], df['CalcDate'].iloc[-1]
    ind = load_industry_data(root_path, scd, ecd, industry_type)
    df = pd.merge(df, ind, how='left', on=['CalcDate', 'Code'])
    rtn = df[~df[industry_type].isin(ind_list)].drop(columns=[industry_type])
    if as_key:
        rtn.set_index(['CalcDate', 'Code'], inplace=True)
    return rtn

    
def get_data_info(root_path, category):
    data_path = get_path(root_path, category)
    files = os.listdir(data_path)
    files = [f for f in files if os.path.splitext(f)[1] == '.ftr']
    files.sort()
    first_file, last_file = files[0], files[-1]
    first_df = pd.read_feather(os.path.join(data_path, first_file))
    first_calc_date = first_df['CalcDate'].iloc[0]
    last_df = pd.read_feather(os.path.join(data_path, last_file))
    last_calc_date = last_df['CalcDate'].iloc[-1]
    rtn = {
        'first_calc_date': first_calc_date,
        'last_calc_date': last_calc_date
    }
    return rtn