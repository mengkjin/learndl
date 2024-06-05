from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd


def resample_trd_calendar(df, date_col, freq_type, **kwargs):
    assert isinstance(df, pd.DataFrame) and isinstance(date_col, str) and date_col in df.columns
    date_list = df[date_col].drop_duplicates().tolist()
    assert date_list and date_list == CALENDAR_UTIL.get_ranged_trading_dates(date_list[0], date_list[-1])
    rsmp_dates = resample_trd_calendar_by_dates(date_list[0], date_list[-1], freq_type, **kwargs)
    rtn = pd.merge(df, pd.DataFrame(rsmp_dates, columns=[date_col]), how='inner', on=[date_col])
    return rtn


def resample_trd_calendar_by_dates(sd, ed, freq_type, **kwargs):
    date_list = CALENDAR_UTIL.get_ranged_trading_dates(sd, ed)
    assert date_list
    if freq_type == 'm' or freq_type == 'M' or freq_type == 'month':
        smp_type = kwargs.pop('smp_type')
        assert smp_type is not None
        rtn = _resample_by_month(date_list, smp_type, **kwargs)
    elif freq_type == 'w' or freq_type == 'W' or freq_type == 'week':
        smp_type = kwargs.pop('smp_type')
        assert smp_type is not None
        rtn = _resample_by_week(date_list, smp_type, **kwargs)
    elif freq_type == 'd' or freq_type == 'D' or freq_type == 'day':
        rtn = date_list
    else:
        assert False, "  error::calendar_tools>>sampling>>resample>>unknown frequency type:{0}.".format(freq_type)
    return rtn


def _resample_by_month(tl, smp_type, **kwargs):
    if smp_type == 'end':
        rtn = CALENDAR_UTIL.get_ranged_eom_trading_dates(tl[0], tl[-1])
    elif smp_type == 'day':
        date = kwargs.get('date')
        assert date is not None
        date = str(date)
        date = date if len(date) == 2 else '0' + date
        rtn = CALENDAR_UTIL.get_last_trading_dates(
            [d[:8] + date for d in CALENDAR_UTIL.get_ranged_eom_trading_dates(tl[0], tl[-1])],
            inc_self_if_is_trdday=True,
            raise_error_if_nan=True
        )

    else:
        raise NotImplementedError
    return rtn


def _resample_by_week(tl, smp_type, **kwargs):
    if smp_type == 'end':
        rtn = CALENDAR_UTIL.get_ranged_eow_trading_dates(tl[0], tl[-1])
    else:
        raise NotImplementedError
    return rtn