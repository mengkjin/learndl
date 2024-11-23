from .wind_conn import get_wind_conn
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
import itertools


def get_st_stocks_from_winddf(start_date_, end_date_):
    assert start_date_ <= end_date_
    sql = "select S_INFO_WINDCODE, ENTRY_DT, REMOVE_DT, ANN_DT, S_TYPE_ST from AShareST " \
          "where ENTRY_DT <= '{0}' and (REMOVE_DT is null or REMOVE_DT >= '{1}')".format(
        end_date_.replace('-', ''), start_date_.replace('-', ''))
    conn = get_wind_conn()
    st_data = pd.read_sql(sql, conn)
    st_data.columns = st_data.columns.str.upper()
    st_data.rename({"S_INFO_WINDCODE": "Code"}, axis=1, inplace=True)
    #
    st_data = st_data[st_data["S_TYPE_ST"] != "R"].copy()
    assert st_data['ENTRY_DT'].notna().all()
    st_data['ENTRY_DT'] = st_data['ENTRY_DT'].str[:4] + '-' + st_data['ENTRY_DT'].str[4:6] + '-' + st_data['ENTRY_DT'].str[6:]
    st_data['last_day_dt'] = st_data['REMOVE_DT'].apply(func=lambda x:
        CALENDAR_UTIL.get_latest_n_dates(x[:4] + '-' + x[4:6] + '-' + x[6:], 2)[0] if x is not None else end_date_)
    st_data = st_data[st_data['last_day_dt'] >= start_date_].copy()
    st_data['first_day_in_prd'] = st_data['ENTRY_DT'].apply(func=lambda x: x if x >= start_date_ else start_date_)
    assert (st_data['first_day_in_prd'] <= st_data['last_day_dt']).all()
    st_data['last_day_dt'] = st_data['last_day_dt'].apply(func=lambda x: x if x <= end_date_ else end_date_)
    st_data['st_dates'] = st_data[['first_day_in_prd', 'last_day_dt']].apply(func=lambda x: CALENDAR_UTIL.get_ranged_dates(x['first_day_in_prd'], x['last_day_dt']), axis=1)
    rtn = st_data[["Code", "st_dates"]].explode('st_dates')
    rtn.drop_duplicates(subset=['Code', 'st_dates'], keep='first', inplace=True)
    rtn.sort_values(by=['st_dates', 'Code'], inplace=True)
    rtn.reset_index(drop=True, inplace=True)
    rtn.rename(columns={'st_dates': 'Date'}, inplace=True)
    rtn = rtn[['Date', 'Code']].copy()
    return rtn


def get_n_listed_dates_from_winddf(start_date, end_date):
    sql = "select S_INFO_WINDCODE, S_INFO_LISTDATE, S_INFO_DELISTDATE " \
          "from AShareDescription where S_INFO_LISTDATE is " \
          "not null and S_INFO_LISTDATE <= '{0}' and " \
          "(S_INFO_DELISTDATE is null or S_INFO_DELISTDATE >= '{1}') order by S_INFO_LISTDATE".format(
        end_date.replace('-', ''), start_date.replace('-', ''))
    conn = get_wind_conn()
    list_delist_data = pd.read_sql(sql, conn)
    list_delist_data.columns = list_delist_data.columns.str.upper()
    list_delist_data.rename(columns={"S_INFO_WINDCODE": "code", "S_INFO_LISTDATE": "listdate", "S_INFO_DELISTDATE": "delistdate"},
                            errors="raise", inplace=True)
    list_delist_data['listdate'] = list_delist_data['listdate'].str[:4] + '-' + list_delist_data['listdate'].str[4:6] + '-' + list_delist_data['listdate'].str[6:]
    list_delist_data['delistdate'] = list_delist_data['delistdate'].apply(func=lambda x: '29991231' if pd.isnull(x) else x)
    list_delist_data['delistdate'] = list_delist_data['delistdate'].str[:4] + '-' + list_delist_data['delistdate'].str[4:6] + '-' + list_delist_data['delistdate'].str[6:]
    dates = CALENDAR_UTIL.get_ranged_dates(start_date, end_date)
    all_data = pd.merge(pd.DataFrame(itertools.product(dates, list_delist_data['code']), columns=['date', 'code']),
             list_delist_data, how='left', on=['code'])
    all_data = all_data[(all_data['date'] < all_data['delistdate']) & (all_data['date'] >= all_data['listdate'])].copy()
    all_data['n_listed_date'] = (pd.to_datetime(all_data['date'].astype(str)) - pd.to_datetime(all_data['listdate'].astype(str))).dt.days + 1
    all_data = all_data[['date', 'code', 'n_listed_date']].rename(columns={'date': 'CalcDate', 'code': 'Code'})
    return all_data


def get_listed_ashare_codes_from_winddf(start_date_, end_date_, without_bj=True, without_delisting=True):
    sql = "select S_INFO_WINDCODE, S_INFO_LISTDATE, S_INFO_DELISTDATE " \
          "from AShareDescription where S_INFO_LISTDATE is " \
          "not null and S_INFO_LISTDATE <= '{0}' and " \
          "(S_INFO_DELISTDATE is null or S_INFO_DELISTDATE >= '{1}') order by S_INFO_LISTDATE".format(
            end_date_.replace('-', ''), start_date_.replace('-', ''))
    conn = get_wind_conn()
    all_data = pd.read_sql(sql, conn)
    all_data.columns = all_data.columns.str.upper()
    all_data.rename(columns={"S_INFO_WINDCODE": "code", "S_INFO_LISTDATE": "listdate", "S_INFO_DELISTDATE": "delistdate"},
                    errors="raise", inplace=True)
    all_data.dropna(subset=['code', 'listdate'], how='any', inplace=True)
    all_data['listdate'] = all_data['listdate'].apply(func=lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    all_data['last_listed_date'] = all_data['delistdate'].apply(
        lambda x: CALENDAR_UTIL.get_latest_n_dates(x[:4] + '-' + x[4:6] + '-' + x[6:], 2)[0] if x is not None else end_date_)
    all_data['last_listed_date'] = all_data['last_listed_date'].apply(func=lambda x: x if x <= end_date_ else end_date_)
    all_data['first_listed_date'] = all_data['listdate'].apply(func=lambda x: x if x >= start_date_ else start_date_)
    all_data['dates'] = all_data[['first_listed_date', 'last_listed_date']].apply(
        func=lambda x: CALENDAR_UTIL.get_ranged_dates(x['first_listed_date'], x['last_listed_date']), axis=1)
    rtn = all_data[['code', 'dates']].explode(column='dates')
    rtn.rename(columns={'dates': 'Date', 'code': 'Code'}, inplace=True)
    rtn.sort_values(by=['Date', 'Code'], inplace=True)
    if without_bj:
        rtn = rtn.loc[rtn['Code'].str[-2:] != 'BJ', ['Date', 'Code']].copy()
    if without_delisting:
        sql = "select S_INFO_WINDCODE, ENTRY_DT from AShareST where S_TYPE_ST in ('T', 'L') and ENTRY_DT <= {0}".format(end_date_.replace('-', ''))
        lt_data = pd.read_sql(sql, conn)
        lt_data.columns = lt_data.columns.str.upper()
        lt_data.rename(columns={"S_INFO_WINDCODE": "Code"}, errors="raise", inplace=True)
        lt_data = lt_data[lt_data['Code'].isin(rtn['Code'].drop_duplicates())].copy()
        lt_data['ENTRY_DT'] = lt_data['ENTRY_DT'].astype(str)
        lt_data['ENTRY_DT'] = lt_data['ENTRY_DT'].str[:4] + '-' + lt_data['ENTRY_DT'].str[4:6] + '-' + \
                              lt_data['ENTRY_DT'].str[6:]
        lt_data['Date'] = lt_data['ENTRY_DT'].apply(lambda x: CALENDAR_UTIL.get_ranged_dates(x, end_date_))
        lt_data = lt_data[['Code', 'Date']].explode('Date')
        lt_data = lt_data[lt_data["Date"].between(start_date_, end_date_)].copy()
        rtn = pd.concat((rtn, lt_data[['Date', 'Code']])).drop_duplicates(subset=['Date', 'Code'],
                                                                          keep=False).sort_values(['Date', 'Code'])
    return rtn


def get_stock_list_n_delist():
    sql = "select S_INFO_WINDCODE, S_INFO_LISTDATE, S_INFO_DELISTDATE from AShareDescription " \
          "where S_INFO_LISTDATE is not null and substr(S_INFO_WINDCODE,1,1) != 'A' order by S_INFO_WINDCODE"
    conn = get_wind_conn()
    rtn = pd.read_sql(sql, conn)
    rtn.columns = rtn.columns.str.upper()
    rtn.rename(columns={"S_INFO_WINDCODE": "code", "S_INFO_LISTDATE": "list_date", "S_INFO_DELISTDATE": "delist_date"},
               errors="raise", inplace=True)
    rtn['delist_date'].fillna(29991231, inplace=True)
    rtn['list_date'] = rtn['list_date'].astype(str)
    rtn['delist_date'] = rtn['delist_date'].astype(str)
    rtn['list_date'] = rtn['list_date'].str[:4] + '-' + rtn['list_date'].str[4:6] + '-' + rtn['list_date'].str[6:8]
    rtn['delist_date'] = rtn['delist_date'].str[:4] + '-' + rtn['delist_date'].str[4:6] + '-' + rtn['delist_date'].str[6:8]
    return rtn


def get_current_stk_name():
    sql = "select S_INFO_WINDCODE, S_INFO_NAME from AShareDescription " \
          "where S_INFO_LISTDATE is not null and substr(S_INFO_WINDCODE,1,1) != 'A'"
    conn = get_wind_conn()
    rtn = pd.read_sql(sql, con=conn)
    rtn.columns = rtn.columns.str.upper()
    rtn.rename({"S_INFO_WINDCODE": "Code", "S_INFO_NAME": "stock_name"}, axis=1, inplace=True)
    return rtn


DISTANT_DATE = "2029-12-31"


def get_stock_description():
    sql = "select S_INFO_WINDCODE, S_INFO_NAME, S_INFO_LISTDATE, " \
          "S_INFO_DELISTDATE, S_INFO_LISTBOARDNAME, S_INFO_COMPCODE " \
          "from AShareDescription " \
          "where S_INFO_LISTDATE is not null and substr(S_INFO_WINDCODE,1,1) != 'A'"
    conn = get_wind_conn()
    rtn = pd.read_sql(sql, con=conn)
    rtn.columns = rtn.columns.str.upper()
    rtn.rename(columns={"S_INFO_WINDCODE": "Code", "S_INFO_NAME": "stock_name", "S_INFO_LISTDATE": "list_date",
                        "S_INFO_DELISTDATE": "delist_date", "S_INFO_LISTBOARDNAME": "list_board", "S_INFO_COMPCODE": "comp_id"},
               errors="raise", inplace=True)
    rtn['delist_date'].fillna(DISTANT_DATE.replace("-", ""), inplace=True)
    rtn['list_date'] = rtn['list_date'].astype(str)
    rtn['delist_date'] = rtn['delist_date'].astype(str)
    rtn['list_date'] = rtn['list_date'].str[:4] + '-' + rtn['list_date'].str[4:6] + '-' + rtn['list_date'].str[6:8]
    rtn['delist_date'] = rtn['delist_date'].str[:4] + '-' + rtn['delist_date'].str[4:6] + '-' + rtn['delist_date'].str[6:8]
    return rtn


def get_stock_st_info():
    sql = "select S_INFO_WINDCODE, ENTRY_DT, REMOVE_DT, ANN_DT, S_TYPE_ST from AShareST"
    conn = get_wind_conn()
    rtn = pd.read_sql(sql, conn)
    rtn.columns = rtn.columns.str.upper()
    rtn.rename(columns={"S_INFO_WINDCODE": "Code", "ENTRY_DT": "entry_dt", "REMOVE_DT": "remove_dt", "ANN_DT": "AnnDate",
                        "S_TYPE_ST": "st_type"},
               errors="raise", inplace=True)
    assert rtn['entry_dt'].notna().all()
    rtn['entry_dt'] = rtn['entry_dt'].str[:4] + '-' + rtn['entry_dt'].str[4:6] + '-' + rtn['entry_dt'].str[6:]
    rtn['remove_dt'].fillna(DISTANT_DATE.replace("-", ""), inplace=True)
    rtn['remove_dt'] = rtn['remove_dt'].str[:4] + '-' + rtn['remove_dt'].str[4:6] + '-' + rtn['remove_dt'].str[6:8]
    return rtn