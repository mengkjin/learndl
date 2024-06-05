from .wind_conn import get_wind_conn
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL


def load_divnsplit_winddf(start_date_, end_date_):
    assert start_date_ <= end_date_
    sql = "select EX_DT, S_INFO_WINDCODE, EQY_RECORD_DT, CASH_DVD_PER_SH_AFTER_TAX, " \
          "STK_DVD_PER_SH, " \
          "S_DIV_BONUSRATE, S_DIV_CONVERSEDRATE, S_DIV_OBJECT from AShareDividend " \
          "where EQY_RECORD_DT > '{0}' and EX_DT <= '{1}' and S_DIV_PROGRESS = '3' " \
          "order by EX_DT, S_INFO_WINDCODE".format(
        start_date_.replace('-', ''),
        end_date_.replace('-', ''))
    conn = get_wind_conn()
    data = pd.read_sql(sql, conn)
    data.columns = data.columns.str.upper()
    data.rename(
        columns={'EX_DT': 'ExDate', 'S_INFO_WINDCODE': 'Code', 'EQY_RECORD_DT': 'RecDate',
                 'CASH_DVD_PER_SH_AFTER_TAX': 'div_rate', 'STK_DVD_PER_SH': 'split_ratio', 'S_DIV_OBJECT': 'div_object'
                 }, errors="raise", inplace=True)
    data['split_ratio'] = data['split_ratio'].fillna(value=0.0)
    data['div_rate'] = data['div_rate'].fillna(value=0.0)
    data['S_DIV_BONUSRATE'] = data['S_DIV_BONUSRATE'].fillna(value=0.0)
    data['S_DIV_CONVERSEDRATE'] = data['S_DIV_CONVERSEDRATE'].fillna(value=0.0)
    #
    data = data[data["div_object"].isin(["普通股股东", "A股流通股"])].copy()
    print("  waring::>>divnsplit>>remove div_object which is not '普通股股东' or 'A股流通股'")
    #
    assert not data['RecDate'].isnull().any()
    assert (data["div_rate"] >= 0).all() and ((
        data["split_ratio"] - data["S_DIV_BONUSRATE"] - data["S_DIV_CONVERSEDRATE"]).abs() < 0.0000001).all()
    data = data[["ExDate", "Code", "RecDate", "div_rate", "split_ratio"]].copy()
    if data.empty:
        data_div_split_ag = pd.DataFrame(columns=["ExDate", "Code", "RecDate", "div_rate", "split_ratio"])
    else:
        data_div_split_ag = data.groupby(by=["ExDate", "Code", "RecDate"], as_index=False)[["div_rate", "split_ratio"]].sum()
    data_div_split_ag['ExDate'] = data_div_split_ag['ExDate'].str[:4] + '-' + \
                                    data_div_split_ag['ExDate'].str[4:6] + '-' + \
                                    data_div_split_ag['ExDate'].str[6:]
    data_div_split_ag['RecDate'] = data_div_split_ag['RecDate'].str[:4] + '-' + \
                                    data_div_split_ag['RecDate'].str[4:6] + '-' + \
                                    data_div_split_ag['RecDate'].str[6:]
    #
    check_dt = data_div_split_ag[['Code', 'ExDate', 'RecDate']].copy()
    trading_dates = CALENDAR_UTIL.get_ranged_trading_dates(start_date_, end_date_)
    if not check_dt['ExDate'].isin(trading_dates).all():
        print("  warning::wind_tools>>divnsplit>>Some ExDates are not trading dates.")
    if not check_dt['RecDate'].isin(trading_dates).all():
        print("  warning::wind_tools>>divnsplit>>Some RecDates are not trading dates.")
    check_dt['next_trddt_after_rec'] = CALENDAR_UTIL.get_next_trading_dates(check_dt['RecDate'].tolist(), False, n=1)
    if not (check_dt['ExDate'] == check_dt['next_trddt_after_rec']).all():
        unusual_record = check_dt.loc[(check_dt['ExDate'] != check_dt['next_trddt_after_rec'])]
        print("  warning::wind_tools>>divnsplit>>Some ExDates are not the next trading dates for RecDates.")
        print(unusual_record)
    return data_div_split_ag


def load_dividend_info(scd, ecd):
    assert scd <= ecd
    sql = "select S_INFO_WINDCODE, REPORT_PERIOD, CASH_DVD_PER_SH_PRE_TAX, " \
          "STK_DVD_PER_SH, S_DIV_BASESHARE, S_DIV_PROGRESS, IS_TRANSFER, "\
          "S_DIV_PRELANDATE, S_DIV_SMTGDATE, DVD_ANN_DT, EX_DT, "\
          "ANN_DT, S_DIV_OBJECT, S_DIV_CHANGE from AShareDividend " \
          "where S_DIV_PRELANDATE > '{0}' and S_DIV_PRELANDATE <= '{1}' " \
          "order by S_DIV_PRELANDATE, S_INFO_WINDCODE".format(scd.replace('-', ''), ecd.replace('-', ''))
    conn = get_wind_conn()
    rtn = pd.read_sql(sql, conn)
    rtn.columns = rtn.columns.str.upper()
    rtn.rename(
        columns={'S_INFO_WINDCODE': 'Code', 'REPORT_PERIOD': 'report_period', 'CASH_DVD_PER_SH_PRE_TAX': 'div_rate',
                 'STK_DVD_PER_SH': 'split_ratio', 'S_DIV_BASESHARE': 'base_share', 'S_DIV_PROGRESS': 'progress',
                 'IS_TRANSFER': 'is_transfer', 'S_DIV_PRELANDATE': 'preplan_date', 'S_DIV_SMTGDATE': 'smtg_date',
                 'DVD_ANN_DT': 'dvd_ann_date', 'EX_DT': 'ExDate', 'ANN_DT': 'latest_ann_date',
                 'S_DIV_OBJECT': 'div_object', 'S_DIV_CHANGE': 'change_description'}, errors="raise", inplace=True)
    rtn["div_rate"] = rtn["div_rate"].fillna(value=0.0)
    assert (rtn["div_rate"] >= 0).all()
    rtn["div_amount"] = rtn["div_rate"] * rtn["base_share"] * 10000.0
    rtn["div_amount"] = rtn["div_amount"].fillna(0.0)
    rtn['report_period'] = rtn["report_period"].astype(str)
    rtn['progress'] = rtn['progress'].astype(str)
    rtn['preplan_date'] = rtn['preplan_date'].str[:4] + '-' + rtn['preplan_date'].str[4:6] + '-' + rtn['preplan_date'].str[6:]
    rtn['smtg_date'] = rtn['smtg_date'].apply(lambda x: "-".join([x[:4], x[4:6], x[6:8]]) if not pd.isnull(x) else None)
    rtn['dvd_ann_date'] = rtn['dvd_ann_date'].apply(lambda x: "-".join([x[:4], x[4:6], x[6:8]]) if not pd.isnull(x) else None)
    rtn['latest_ann_date'] = rtn['latest_ann_date'].apply(
        lambda x: "-".join([x[:4], x[4:6], x[6:8]]) if not pd.isnull(x) else None)
    rtn['ExDate'] = rtn['ExDate'].apply(lambda x: "-".join([x[:4], x[4:6], x[6:8]]) if not pd.isnull(x) else None)
    return rtn