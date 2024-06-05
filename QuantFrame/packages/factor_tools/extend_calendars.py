import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL


def extend_trade_to_calendars(data, ecd):
    assert pd.Index(["CalcDate", "Code"]).difference(data.columns).empty
    assert data["CalcDate"].is_monotonic
    assert set(data["CalcDate"]).issubset(
        CALENDAR_UTIL.get_ranged_trading_dates(data["CalcDate"].iloc[0], data["CalcDate"].iloc[-1]))
    data = data.rename(columns={"CalcDate": "trade_CalcDate"}, errors="raise")
    date_df = data[["trade_CalcDate"]].drop_duplicates()
    date_df["calendar_dates"] = date_df["trade_CalcDate"].apply(
        lambda x: CALENDAR_UTIL.get_ranged_dates(
            x, CALENDAR_UTIL.get_next_trading_dates([x], inc_self_if_is_trdday=False)[0])[:-1])
    rtn = pd.merge(data, date_df, on=["trade_CalcDate"], how="left").rename(
        columns={"calendar_dates": "CalcDate"}, errors="raise")
    rtn = rtn.explode("CalcDate").drop(columns=["trade_CalcDate"]).\
        sort_values(["CalcDate", "Code"])
    rtn = rtn[pd.Index(["CalcDate", "Code"]).append(rtn.columns.drop(["CalcDate", "Code"]))]
    return rtn


def extend_data_to_calendar_days(data, ecd=None, calendar_type='calendar'):
    assert pd.Index(["CalcDate", "Code"]).difference(data.columns).empty
    assert data["CalcDate"].is_monotonic_increasing
    if ecd is None:
        ecd = data["CalcDate"].iloc[-1]
    assert ecd >= data["CalcDate"].iloc[-1]
    date_df = data[["CalcDate"]].drop_duplicates().reset_index(drop=True)
    date_df["next_CalcDate"] = date_df["CalcDate"].shift(-1)
    date_df["next_CalcDate"].mask(date_df["CalcDate"] == date_df["CalcDate"].iloc[-1],
                                  CALENDAR_UTIL.get_next_dates([ecd])[0], inplace=True)
    date_df["fill_dates"] = date_df.apply(
        lambda x: CALENDAR_UTIL.get_ranged_dates(x["CalcDate"], x["next_CalcDate"])[:-1], axis=1)
    rtn = pd.merge(data, date_df[["CalcDate", "fill_dates"]], how="inner", on=["CalcDate"]).\
        drop(columns=["CalcDate"]).rename(columns={"fill_dates": "CalcDate"}, errors="raise")
    rtn = rtn.explode("CalcDate").sort_values(["CalcDate", "Code"])
    rtn = rtn[pd.Index(["CalcDate", "Code"]).append(rtn.columns.drop(["CalcDate", "Code"]))].copy()
    return rtn


def extend_data_to_calendar_days_new(data, ecd=None, calendar_type='full', date_code_cols=None):
    if date_code_cols is None:
        assert isinstance(data.index, pd.MultiIndex) and len(data.index.names) == 2
        date_col, code_col = data.index.names
        data = data.reset_index()
    else:
        date_col, code_col = date_code_cols
    assert data[date_col].is_monotonic_increasing
    if ecd is None:
        ecd = data[date_col].iloc[-1]
    assert ecd >= data[date_col].iloc[-1]
    date_df = data[[date_col]].drop_duplicates().reset_index(drop=True)
    date_df["next_date"] = date_df[date_col].shift(-1)
    date_df["next_date"].mask(date_df[date_col] == date_df[date_col].iloc[-1],
                                  CALENDAR_UTIL.get_next_dates([ecd])[0], inplace=True)
    date_df["fill_dates"] = date_df.apply(
        lambda x: CALENDAR_UTIL.get_ranged_dates(x[date_col], x["next_date"])[:-1], axis=1)
    rtn = pd.merge(data, date_df[[date_col, "fill_dates"]], how="left", on=[date_col]).\
        drop(columns=[date_col]).rename(columns={"fill_dates": date_col}, errors="raise")
    rtn = rtn.explode(date_col).sort_values([date_col, code_col])
    rtn = rtn[pd.Index([date_col, code_col]).append(rtn.columns.drop([date_col, code_col]))].copy()
    if date_code_cols is None:
        rtn.set_index([date_col, code_col], inplace=True)
    return rtn
