import pandas as pd
from .config import get_file_path
from events_system.calendar_util import CALENDAR_UTIL


def load_stk_basic_info_data(root_path, category):
    path = get_file_path(root_path, category)
    rtn = pd.read_csv(path, encoding="gbk")
    return rtn


def get_n_listed_dates(root_path, scd, ecd):
    data = load_stk_basic_info_data(root_path, "description")
    data = data.loc[(data["list_date"] <= ecd) & (data["delist_date"] >= scd), ["Code", "list_date", "delist_date"]].copy()
    data["list_end_date"] = data["delist_date"].apply(lambda x: CALENDAR_UTIL.get_latest_n_dates(x, 2)[0] if x <= ecd else ecd)
    data["CalcDate"] = data.apply(lambda x: CALENDAR_UTIL.get_ranged_dates(max(x["list_date"], scd), x["list_end_date"]), axis=1)
    all_data = data[["CalcDate", "Code", "list_date"]].explode(column="CalcDate")
    all_data['n_listed_date'] = (pd.to_datetime(all_data['CalcDate'], format="%Y-%m-%d") - pd.to_datetime(all_data['list_date'], format="%Y-%m-%d")).dt.days + 1
    all_data = all_data[['CalcDate', 'Code', 'n_listed_date']].copy()
    return all_data


def get_listed_ashare_codes(root_path, scd, ecd, without_bj=True, without_delisting=True):
    data = load_stk_basic_info_data(root_path, "description")
    data = data.loc[(data["list_date"] <= ecd) & (data["delist_date"] >= scd), ["Code", "list_date", "delist_date"]].copy()
    #
    if without_bj:
        data = data.loc[data['Code'].str[-2:] != 'BJ'].copy()
    #
    data["list_end_date"] = data["delist_date"].apply(lambda x: CALENDAR_UTIL.get_latest_n_dates(x, 2)[0] if x <= ecd else ecd)
    data["Date"] = data.apply(lambda x: CALENDAR_UTIL.get_ranged_dates(max(x["list_date"], scd), x["list_end_date"]), axis=1)
    rtn = data[["Date", "Code"]].explode(column="Date")
    if without_delisting:
        lt_data = load_stk_basic_info_data(root_path, "st")
        lt_data = lt_data[lt_data["st_type"].isin(['T', 'L']) & (lt_data["entry_dt"] <= ecd)].copy()
        lt_data = lt_data[lt_data['Code'].isin(rtn['Code'].unique())].copy()
        lt_data['Date'] = lt_data['entry_dt'].apply(lambda x: CALENDAR_UTIL.get_ranged_dates(max(scd, x), ecd))
        lt_data = lt_data[['Code', 'Date']].explode('Date')
        rtn = pd.concat((rtn, lt_data[['Date', 'Code']]), axis=0)
        rtn.drop_duplicates(subset=['Date', 'Code'], keep=False, inplace=True)
        rtn.sort_values(['Date', 'Code'], inplace=True)
    return rtn


def get_st_stocks(root_path, scd, ecd):
    data = load_stk_basic_info_data(root_path, "st")
    data = data[data["st_type"] != "R"].copy()
    data = data.loc[(data["entry_dt"] <= ecd) & (data["remove_dt"] >= scd), ["Code", "entry_dt", "remove_dt"]].copy()
    data["st_end_date"] = data["remove_dt"].apply(lambda x: CALENDAR_UTIL.get_latest_n_dates(x, 2)[0] if x <= ecd else ecd)
    data["Date"] = data.apply(lambda x: CALENDAR_UTIL.get_ranged_dates(max(x["entry_dt"], scd), x["st_end_date"]), axis=1)
    rtn = data[["Date", "Code"]].explode('Date')
    rtn.drop_duplicates(subset=['Date', 'Code'], keep='first', inplace=True)
    rtn = rtn[['Date', 'Code']].sort_values(by=['Date', 'Code']).reset_index(drop=True)
    return rtn


def get_early_delist(root_path, scd, ecd):
    df = load_stk_basic_info_data(root_path, "description")
    df = df.loc[(df["delist_date"] >= scd) & (df["early_delist_date"] <= ecd), ["Code", "early_delist_date", "delist_date"]].copy()
    df['Date'] = df.apply(lambda x: CALENDAR_UTIL.get_ranged_dates(x['early_delist_date'], x['delist_date']), axis=1)
    rtn = df[['Code', 'Date']].explode('Date')
    rtn = rtn.loc[rtn['Date'].between(scd, ecd), ['Date', 'Code']].sort_values(['Date', 'Code'])
    rtn['remove_by_delist'] = 1
    return rtn


def get_stock_list_n_delist(root_path):
    rtn = load_stk_basic_info_data(root_path, "description")
    rtn = rtn[["Code", "list_date", "delist_date"]].copy()
    return rtn


def get_current_stk_name(root_path):
    rtn = load_stk_basic_info_data(root_path, "description")
    rtn = rtn[["Code", "stock_name"]].copy()
    return rtn


def load_ashare_comp_id(root_path):
    rtn = load_stk_basic_info_data(root_path, "description")
    rtn = rtn[["Code", "comp_id"]].copy()
    return rtn