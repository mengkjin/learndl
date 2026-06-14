"""
Trading-calendar utilities: natural dates (cd) vs trading dates (td), ranges, diffs, and quarter helpers.

Uses ``BasicCalendar`` (``BC``) for fast ndarray-backed lookups. Prefer ``CALENDAR`` and ``TradeDate``
from this package for all research code.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Any, Iterable

from src.proj.core import NoInstanceMeta , lit , as_int_array
from src.proj.env import PATH , Const

from .basic import (
    BJ_TZ , BC , TradeDate , intDate , intDateNone , intDates , get_cd , get_td)

__all__ = ['CALENDAR']

class CALENDAR(metaclass=NoInstanceMeta):
    """Static tools for trading date and natural date conversion, interval truncation, quarter end, etc."""

    _ME : np.ndarray | Any = None
    _QE : np.ndarray | Any = None
    _YE : np.ndarray | Any = None
    _update_to : int | None = None

    @classmethod
    def ensure_period_ends(cls):
        if cls._ME is None:
            cls._ME = pd.date_range(start="1997-01-01", end="2099-12-31", freq="ME").strftime("%Y%m%d").to_numpy(int)
        if cls._QE is None:
            cls._QE = pd.date_range(start="1997-01-01", end="2099-12-31", freq="QE").strftime("%Y%m%d").to_numpy(int)
        if cls._YE is None:
            cls._YE = pd.date_range(start="1997-01-01", end="2099-12-31", freq="YE").strftime("%Y%m%d").to_numpy(int)

    @staticmethod
    def now(bj_tz: bool = True) -> datetime:
        """Current time; use 'BJTZ' (Shanghai) if 'bj_tz' is True."""
        tz = BJ_TZ if bj_tz else None
        return datetime.now(tz)

    @classmethod
    def today(cls, offset=0, bj_tz=True) -> int:
        """Today's natural date 'YYYYMMDD'; 'offset' is the number of days offset from today."""
        d = cls.now(bj_tz)
        if offset != 0:
            d = d + timedelta(days=offset)
        return int(d.strftime("%Y%m%d"))

    @classmethod
    def time(cls, bj_tz=True) -> int:
        """Current time 'HHMMSS' as an integer."""
        return int(cls.now(bj_tz).strftime("%H%M%S"))

    @staticmethod
    def datetime(year: int = 2000, month: int = 1, day: int = 1, *, lock_month=False) -> int:
        """
        Combine year, month and day to form the natural date 'YYYYMMDD'; 
        - the 'month' and 'day' can overflow;
        - 'lock_month' controls the month end semantics.
        """
        if month > 12 or month <= 0:
            year += (month - 1) // 12
            month = (month - 1) % 12 + 1
        date1 = datetime(year, month, 1)
        real_month = date1.month if lock_month else None
        date = date1 + timedelta(days=day - 1)
        if real_month is not None and date.month != real_month:
            if date.month < real_month:
                date = date1.replace(day=1)
            else:
                date = date1.replace(month=real_month + 1, day=1) - timedelta(days=1)
        return int(date.strftime("%Y%m%d"))

    @classmethod
    def clock(cls, date: int = 0, h=0, m=0, s=0):
        """
        Combine the date and time to form the 'YYYYMMDDHHMMSS'; 
        - relative to update_to if 'date<=0'.
        """
        if date <= 0:
            date_time = cls.now(bj_tz=False) + timedelta(days=date)
        else:
            date_time = datetime.strptime(str(date), "%Y%m%d")
        return int(date_time.replace(hour=h, minute=m, second=s).strftime("%Y%m%d%H%M%S"))

    @staticmethod
    def bjtime_to_localtime(bj_time: int | float) -> int:
        """Convert BJ time to local time , in and out are of format 'YYYYMMDDHHMMSS'"""
        bj_datetime = datetime.strptime(str(int(bj_time)), "%Y%m%d%H%M%S").replace(tzinfo = BJ_TZ)
        local_datetime = bj_datetime.astimezone(None)
        return int(local_datetime.strftime("%Y%m%d%H%M%S"))

    @staticmethod
    def localtime_to_bjtime(local_time: int | float) -> int:
        """Convert local time to BJ time , in and out are of format 'YYYYMMDDHHMMSS'"""
        local_datetime = datetime.strptime(str(int(local_time)), "%Y%m%d%H%M%S").replace(tzinfo = None)
        bj_datetime = local_datetime.astimezone(BJ_TZ)
        return int(bj_datetime.strftime("%Y%m%d%H%M%S"))

    @classmethod
    def get_modified_time(
        cls, file_or_modified_time: Iterable[Path] | Path | int | float | None , * , 
        bj_tz: bool = True
    ) -> int:
        if isinstance(file_or_modified_time, Iterable):
            mtime = min([PATH.file_modified_time(path) for path in file_or_modified_time])
        elif isinstance(file_or_modified_time, Path):
            mtime = PATH.file_modified_time(file_or_modified_time)
        elif isinstance(file_or_modified_time, int | float):
            mtime = int(file_or_modified_time)
        else:
            mtime = 19970101000000
        if mtime < 1e8:
            mtime = mtime * 1000000
        if bj_tz:
            mtime = cls.localtime_to_bjtime(mtime)
        return mtime

    @classmethod
    def is_updated_today(
        cls, file_or_modified_time: Iterable[Path] | Path | int | float | None, 
        hour=20, minute=0 , * , bj_tz: bool = True
    ) -> bool:
        """
        Check if 'modified_time' is not earlier than the corresponding time of "required_date + hour:minute".
        modified_time can be 'YYYYMMDD' or 'YYYYMMDDHHMMSS' or 'YYYYMMDDHHMMSS.MS'.
        """
        modified_time = cls.get_modified_time(file_or_modified_time , bj_tz=bj_tz)
        bjtime = cls.now(bj_tz=bj_tz)
        if int(bjtime.strftime("%H%M")) < hour * 100 + minute:
            bjtime = bjtime - timedelta(days=1)
        required_time = cls.clock(int(bjtime.strftime("%Y%m%d")), hour, minute)
        return modified_time >= required_time

    @classmethod
    def is_updated_recently(
        cls, file_or_modified_time: Iterable[Path] | Path | int | float | None, 
        hours : float = 1. , * , bj_tz: bool = True
    ) -> bool:
        """
        Check if 'modified_time' is not earlier than the corresponding time of "required_date + hour:minute".
        modified_time can be 'YYYYMMDD' or 'YYYYMMDDHHMMSS' or 'YYYYMMDDHHMMSS.MS'.
        """
        modified_time = cls.get_modified_time(file_or_modified_time , bj_tz=bj_tz)
        bjtime = cls.now(bj_tz=bj_tz)
        shift_time = bjtime - timedelta(hours=hours)
        shift_time = int(shift_time.strftime("%Y%m%d%H%M%S"))
        return modified_time >= shift_time

    @classmethod
    def update_from(cls , key: lit.DataUpdateKey | None = None) -> int:
        """update to date"""
        return Const.Data.UPDATE.start(key)

    @classmethod
    def update_to(cls , offset: int = 0 , key: lit.DataUpdateKey | None = None) -> int:
        """
        The cached 'updated to' natural date; 
        - take today or yesterday based on whether the current time is after 19:59.
        if offset != 0, the update_to will be truncated to today + offset.
        at platform coding, the update_to will be the latest date in the 'trade_ts/day' database table.
        """

        if cls._update_to is None:
            cls._update_to = cls.today(-1 if cls.now(bj_tz=True).time() <= time(19, 59, 0) else 0)
        update_to = min(cls._update_to, Const.Data.UPDATE.end(key))
        update_to = cls.cd(update_to , offset)
        return update_to

    @classmethod
    def update_schedule(
        cls, start: intDateNone = None, end: intDateNone = None , 
        key: lit.DataUpdateKey | None = None
    ) -> tuple[int, int]:
        """trim the start and end date of the update schedule"""
        start = 19000101 if start is None else get_cd(start)
        start = max(start, cls.update_from(key))
        
        end = 99991231 if end is None else get_cd(end)
        end = min(end, cls.update_to(key=key))
        return start , end

    @staticmethod
    def updated(date: intDateNone = None) -> int:
        """The maximum date in the 'trade_ts/day' database table; optionally truncated by 'date'."""
        from src.proj.db import DB
        updated_date = DB.max_date("trade_ts", "day")
        if date is not None:
            updated_date = min(updated_date, int(date))
        return updated_date

    @classmethod
    def offset(cls, dates: intDates | None, offset: int = 0, type: lit.intDateType = 'td' , backward=True) -> np.ndarray[Any, np.dtype[np.int_]]:
        """Convert multiple natural dates to trading dates (or 'td_forward'); optionally offset by 'offset' steps."""
        if dates is None:
            return np.array([], dtype=np.int64)
        return BC.offset_np(as_int_array(dates), offset, type, backward)

    @classmethod
    def td(cls , date: intDate, offset: int = 0 , backward=True) -> TradeDate:
        """Return 'TradeDate'; optionally offset by 'offset' steps."""
        if backward:
            return TradeDate(date).offset(offset)
        else:
            return TradeDate(cls.tds(date, backward=False)[0])

    @classmethod
    def tds(cls, dates: intDates, backward=True) -> np.ndarray:
        """Convert multiple natural dates to trading dates (or 'td_forward'); optionally offset by 'offset' steps."""
        return BC.offset_np(dates, 0, 'td', backward)

    @staticmethod
    def cd(date: intDate, offset: int = 0) -> int:
        """Return the natural date 'YYYYMMDD'; optionally offset by 'offset' steps."""
        d = get_cd(date)
        if offset == 0 or d <= BC.min_date or d >= BC.max_date:
            return d
        d = datetime.strptime(str(d), "%Y%m%d") + timedelta(days=offset)
        return int(d.strftime("%Y%m%d"))

    @classmethod
    def cds(cls, dates: intDates) -> np.ndarray:
        """Convert multiple trading dates to natural dates; optionally offset by 'offset' steps."""
        return as_int_array(dates)

    @classmethod
    def diff_days(cls, date1 : intDate, date2 : intDate , type : lit.intDateType = 'td') -> int | Any:
        """The difference in days between two dates."""
        return BC.diff_days(date1, date2, type)

    @classmethod
    def diff_days_array(cls, date1_arr : intDates, date2_arr : intDates , type : lit.intDateType = 'td') -> int | Any:
        """The difference of two arrays of dates."""
        return BC.diff_days_array(date1_arr, date2_arr, type)

    @classmethod
    def trailing(cls, date : intDate, n: int, type: lit.intDateType = 'td') -> np.ndarray[Any, np.dtype[np.int_]]:
        """The last 'n' trading days before 'date' (sorted slice)."""
        return BC.trailing_np(date, n, type)

    @classmethod
    def as_start_date(cls, date: intDateNone) -> int:
        """clear the start date of input date (None -> 19000101, negative -> relative to update_to)"""
        date_dt = 19000101 if date is None else get_cd(date)
        if date_dt < 0:
           date_dt = cls.update_to(date_dt)
        return date_dt

    @classmethod
    def as_end_date(cls, date: intDateNone) -> int:
        """clear the end date of input date (None -> 99991231, negative -> relative to update_to)"""
        date = 99991231 if date is None else get_cd(date)
        if date < 0:
            date = cls.update_to(date)
        return date

    @classmethod
    def range(
        cls, start: intDateNone, end: intDateNone , type : lit.intDateType = 'td' ,
        step: int = 1 , until_today=True, updated=False,
    ) -> np.ndarray[Any, np.dtype[np.int_]]:
        """return the range of dates between start and end"""
        dates = BC.range(start, end, type)
        if until_today:
            dates = dates[dates <= cls.update_to()]
        if updated:
            dates = dates[dates <= cls.updated()]
        dates = dates[::step]
        return dates

    @staticmethod
    def calendar_start():
        """The minimum natural date in the calendar."""
        return BC.min_date

    @staticmethod
    def calendar_end():
        """The maximum natural date in the calendar."""
        return BC.max_date

    @classmethod
    def td_start_end(
        cls , reference_date : intDate,
        period_num: int = 1, freq: lit.FreqPeriod | str = "m", lag_num: int = 0,
    ) -> tuple[int, int]:
        """Calculate the start and end date of a trading date interval based on the approximate trading date length (d/w/m/q/y) from the reference date."""
        pdays = {"d": 1, "w": 7, "m": 21, "q": 63, "y": 252}[freq]
        start = cls.td(reference_date , - pdays * (period_num + lag_num) + 1).as_int()
        end = cls.td(reference_date , - pdays * lag_num).as_int()
        return start, end

    @classmethod
    def cd_start_end(
        cls, reference_date : intDate,
        period_num: int = 1, freq: lit.FreqPeriod | str = "m", lag_num: int = 0,
    ) -> tuple[int, int]:
        """Calculate the start and end date of a natural date interval based on the approximate natural date length (d/w/m/q/y) from the reference date."""
        if freq in ["d", "w"]:
            pdays = {"d": 1, "w": 7}[freq]
            start = cls.cd(reference_date , - pdays * (period_num + lag_num) + 1)
            end = cls.cd(reference_date , - pdays * lag_num)
        elif freq in ["m", "q", "y"]:
            cd = get_cd(reference_date)
            year, month, day = cd // 10000, cd % 10000 // 100, cd % 100
            pmonths = {"m": 1, "q": 3, "y": 12}[freq]
            start_month = month - pmonths * (period_num + lag_num)
            end_month = month - pmonths * lag_num
            start = cls.cd(cls.datetime(year, start_month, day, lock_month=True), 1)
            end = cls.datetime(year, end_month, day, lock_month=True)
        else:
            raise ValueError(f"Invalid frequency: {freq}")
        return start, end

    @staticmethod
    def is_trade_date(date: intDate) -> bool:
        """Whether the date is a trading day; 'int(TradeDate)' is 'td', participate in the judgment."""
        return True if isinstance(date, TradeDate) else BC.is_trade_cd(date)

    @staticmethod
    def trade_dates():
        """All trading dates 'YYYYMMDD' in ascending order (copy)."""
        return BC._trade_calendar.copy()

    @staticmethod
    def reformat(date: Any, old_fmt = "%Y%m%d", new_fmt : str | None = "%Y%m%d"):
        """Convert the date string format; return the original string if the two formats are the same."""
        if old_fmt != new_fmt and new_fmt is not None:
            return datetime.strptime(str(date), old_fmt).strftime(new_fmt)
        else:
            return date

    @classmethod
    def year_end(cls, date: intDate):
        """The last day of the year (December 31) of the year the date is in ('YYYYMMDD')."""
        return get_td(date) // 10000 * 10000 + 1231

    @classmethod
    def year_start(cls, date: intDate):
        """The first day of the year (January 1) of the year the date is in ('YYYYMMDD')."""
        return get_td(date) // 10000 * 10000 + 101

    @classmethod
    def quarter_end(cls, date: intDate):
        """The last day of the quarter."""
        cls.ensure_period_ends()
        return cls._QE[cls._QE >= get_td(date)][0]

    @classmethod
    def quarter_start(cls, date: intDate):
        """The first day of the quarter."""
        td = get_td(date)
        return td // 10000 * 10000 + (td % 10000 // 100 / 3).__floor__() * 300 + 101

    @classmethod
    def month_end(cls, date: intDate):
        """The last day of the month."""
        cls.ensure_period_ends()
        return cls._ME[cls._ME >= get_td(date)][0]

    @classmethod
    def month_start(cls, date: intDate):
        """The first day of the month."""
        return get_td(date) // 100 * 100 + 1

    @classmethod
    def qe_trailing(cls, date: intDate, n_past=1, n_future=0, another_date: intDateNone = None, year_only=False):
        """
        Calculate the trailing quarter ends (or year ends) around 'date' and 'another_date'; optionally calculate the future 'n_future' quarter ends.
        
        inputs:
            date: the date to calculate the trailing quarter ends
            n_past: the number of past quarter ends to calculate
            n_future: the number of future quarter ends to calculate
            another_date: if supplied, all quarter/year ends between will be included in the result.
            year_only: whether to calculate the trailing year ends
        """
        assert n_past >= 1 and n_future >= 0, f"{n_past} and {n_future} must be greater than 1 and 0"
        cls.ensure_period_ends()
        dates = cls._YE if year_only else cls._QE
        d0 = get_td(date)
        d1 = get_td(another_date) if another_date is not None else d0
        d0 , d1 = min(d0, d1), max(d0, d1)
        before = dates[dates <= d0][-n_past:]
        between = dates[(dates > d0) & (dates <= d1)]
        after = dates[dates > d1][:n_future]
        return np.concatenate([before, between, after])

    @classmethod
    def qe_within(cls, start : intDate, end : intDate, year_only : bool = False):
        """The set of quarter ends (or year ends) within the interval. optionally can only include year ends."""
        cls.ensure_period_ends()
        if year_only:
            return cls._YE[(cls._YE >= get_td(start)) & (cls._YE <= get_td(end))]
        return cls._QE[(cls._QE >= get_td(start)) & (cls._QE <= get_td(end))]

    @classmethod
    def qe_interpolate(cls, incomplete_qtr_ends: intDates):
        """Expand the minimum and maximum range of several quarter ends to a complete quarter end sequence."""
        qes = as_int_array(incomplete_qtr_ends)
        if len(qes) == 0:
            return qes
        return cls.qe_within(min(qes), max(qes))

    @classmethod
    def check_rollback_date(cls, rollback_date: intDateNone, max_rollback_days: int = 10):
        """Assert that 'rollback_date' is not earlier than max_rollback_days days before the updated date."""
        if rollback_date is None:
            return
        earliest_rollback_date = cls.td(cls.updated(), -max_rollback_days)
        assert get_td(rollback_date) >= earliest_rollback_date, (
            f"rollback_date {rollback_date} is too early, must be at least {earliest_rollback_date}"
        )