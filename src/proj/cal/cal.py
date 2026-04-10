"""
Trading-calendar utilities: natural dates (cd) vs trading dates (td), ranges, diffs, and quarter helpers.

Uses ``BasicCalendar`` (``BC``) for fast ndarray-backed lookups. Prefer ``CALENDAR`` and ``TradeDate``
from this package for all research code.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Any, Literal, Union , Sequence , Iterable

from src.proj.core import NoInstanceMeta
from src.proj.env import PATH
import src.proj.db as DB

from .basic import BasicCalendar , BJ_TZ
from .trade_date import TradeDate

__all__ = ['CALENDAR', 'Dates']

BC = BasicCalendar()

DateType = Union[int, TradeDate]
DateTypeWithNone = Union[int, TradeDate, None]
DatesType = Union[DateType , Sequence[DateType], np.ndarray , pd.Series , torch.Tensor]

def get_cd(date: DateType) -> int:
    """Natural calendar day ``YYYYMMDD`` as int; for ``TradeDate``, returns ``.cd``."""
    return int(date.cd if isinstance(date, TradeDate) else date)


def get_td(date: DateType) -> int:
    """Trading-aligned day ``YYYYMMDD`` as int; for ``TradeDate``, returns ``.td``."""
    return int(date.td if isinstance(date, TradeDate) else date)


def get_cds(dates: DatesType) -> np.ndarray:
    """
    Convert the 'date' parameter of 'td_array' / '*_diff_array' to an integer array index key.
    Single element of this package 'TradeDate' will be expanded to its 'td'; 'pd.Series' will be converted to 'ndarray'.
    """
    if isinstance(dates, (TradeDate , int)):
        dates = np.array([get_cd(dates)])
    elif isinstance(dates, pd.Series):
        dates = dates.to_numpy(int)
    elif isinstance(dates, np.ndarray):
        ...
    elif isinstance(dates, Sequence):
        dates = np.array([get_cd(d) for d in dates])
    elif isinstance(dates , torch.Tensor):
        dates = dates.cpu().numpy()
    return dates.astype(int)

class CALENDAR(metaclass=NoInstanceMeta):
    """Static tools for trading date and natural date conversion, interval truncation, quarter end, etc."""

    ME = pd.date_range(start="1997-01-01", end="2099-12-31", freq="ME").strftime("%Y%m%d").to_numpy(int)
    QE = pd.date_range(start="1997-01-01", end="2099-12-31", freq="QE").strftime("%Y%m%d").to_numpy(int)
    YE = pd.date_range(start="1997-01-01", end="2099-12-31", freq="YE").strftime("%Y%m%d").to_numpy(int)

    _update_to = None

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
        - relative to today if 'date<=0'.
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
    def get_modified_time(cls, file_or_modified_time: Iterable[Path] | Path | int | float | None , * , bj_tz: bool = True) -> int:
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
    def is_updated_today(cls, file_or_modified_time: Iterable[Path] | Path | int | float | None, hour=20, minute=0 , * , bj_tz: bool = True):
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
    def is_updated_recently(cls, file_or_modified_time: Iterable[Path] | Path | int | float | None, hours : float = 1. , * , bj_tz: bool = True):
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
    def update_to(cls):
        """The cached "updated to" natural date; take today or yesterday based on whether the current time is after 19:59."""
        if cls._update_to is None:
            cls._update_to = cls.today(-1 if cls.now(bj_tz=True).time() <= time(19, 59, 0) else 0)
        return cls._update_to

    @staticmethod
    def updated(date: DateTypeWithNone = None):
        """The maximum date in the 'trade_ts/day' database table; optionally truncated by 'date'."""
        updated_date = DB.max_date("trade_ts", "day")
        if date is not None:
            updated_date = min(updated_date, int(date))
        return updated_date

    @staticmethod
    def td(date: DateType, offset: int = 0):
        """Return 'TradeDate'; optionally offset by 'offset' steps."""
        return TradeDate(date).offset(offset)

    @classmethod
    def td_array(cls, dates: DatesType | None, offset: int = 0, backward=True):
        """Convert multiple natural dates to trading dates (or 'td_forward'); optionally offset by 'offset' steps."""
        if dates is None:
            return np.array([], dtype=np.int64)
        pos = BC.pos_cd_array(get_cds(dates))
        td_arr = (BC._td_col if backward else BC._td_forward)[pos]
        td_arr = np.asarray(td_arr, dtype=np.int64)
        if offset != 0:
            pos_td = BC.pos_cd_array(td_arr)
            d_index = BC._td_index[pos_td] + offset
            d_index = np.maximum(np.minimum(d_index, BC.n_td - 1), 0)
            td_arr = BC.trade_calendar_by_td_index(d_index)
            td_arr = np.asarray(td_arr, dtype=np.int64)
        return td_arr

    @staticmethod
    def cd(date: DateType, offset: int = 0):
        """Return the natural date 'YYYYMMDD'; optionally offset by 'offset' steps."""
        d = get_cd(date)
        if offset == 0 or d <= BC.min_date or d >= BC.max_date:
            return d
        d = datetime.strptime(str(d), "%Y%m%d") + timedelta(days=offset)
        return int(d.strftime("%Y%m%d"))

    @staticmethod
    def cd_array(dates: DatesType | None, offset: int = 0):
        """Shift the natural date index of the sequence; the elements must be 'TradeDate' or natural dates that can be 'int'."""
        if dates is None:
            return np.array([], dtype=np.int64)
        cd_arr = get_cds(dates)
        if offset != 0:
            cd_arr = np.minimum(cd_arr, CALENDAR.calendar_end())
            d_index = BC._cd_index[BC.pos_cd_array(cd_arr)] + offset
            d_index = np.maximum(np.minimum(d_index, BC.n_cal - 1), 0)
            cd_arr = np.asarray(BC.calendar_by_cd_index(d_index), dtype=np.int64)
        return cd_arr

    @staticmethod
    def td_diff(date1 : DateType, date2 : DateType) -> int | Any:
        """The difference in trading date index between two dates."""
        try:
            i1 = BC.pos_cd(get_cd(date1))
            i2 = BC.pos_cd(get_cd(date2))
            diff = int(BC._td_index[i2] - BC._td_index[i1])
        except KeyError:
            diff = None
        assert isinstance(diff, int), f"{date1} and {date2} are not in calendar"
        return diff

    @staticmethod
    def cd_diff(date1 : DateType, date2 : DateType) -> int | Any:
        """The difference in natural date index between two dates"""
        try:
            i1 = BC.pos_cd(get_cd(date1))
            i2 = BC.pos_cd(get_cd(date2))
            diff = int(BC._cd_index[i2] - BC._cd_index[i1])
        except KeyError:
            diff = (datetime.strptime(str(date1), "%Y%m%d") - datetime.strptime(str(date2), "%Y%m%d")).days
        return diff

    @classmethod
    def td_diff_array(cls, date1_arr : DatesType, date2_arr : DatesType) -> int | Any:
        """The difference of two arrays of trading date."""
        p1 = BC.pos_cd_array(get_cds(date1_arr))
        p2 = BC.pos_cd_array(get_cds(date2_arr))
        return BC._td_index[p1] - BC._td_index[p2]

    @classmethod
    def cd_diff_array(cls, date1_arr : DatesType, date2_arr : DatesType) -> int | Any:
        """The difference of two arrays of natural dates."""
        p1 = BC.pos_cd_array(get_cds(date1_arr))
        p2 = BC.pos_cd_array(get_cds(date2_arr))
        return BC._cd_index[p1] - BC._cd_index[p2]

    @staticmethod
    def td_trailing(date : DateType, n: int):
        """The last 'n' trading days before 'date' (sorted slice)."""
        return np.sort(BC.td_trailing_np(get_td(date), n)).astype(int)

    @staticmethod
    def cd_trailing(date : DateType, n: int):
        """The last 'n' natural days before 'date' (sorted slice)."""
        return np.sort(BC.cd_trailing_np(get_cd(date), n)).astype(int)

    @classmethod
    def as_start_date(cls, date: DateTypeWithNone) -> int:
        """clear the start date of input date (None -> 19900101, negative -> relative to today)"""
        date_dt = 19900101 if date is None else get_cd(date)
        #if date_dt < 0:
        #    date_dt = cls.today(date_dt)
        return date_dt

    @classmethod
    def as_end_date(cls, date: DateTypeWithNone) -> int:
        """clear the end date of input date (None -> 99991231, negative -> relative to today)"""
        date = 99991231 if date is None else get_cd(date)
        if date < 0:
            date = cls.today(date)
        return date

    @classmethod
    def range(
        cls, start: DateTypeWithNone, end: DateTypeWithNone , type : Literal['td', 'cd'] = 'td' ,
        step: int = 1 , until_today=True, updated=False,
        slice: tuple[DateTypeWithNone, DateTypeWithNone] | None = None,
    ) -> np.ndarray[int, Any]:
        """return the range of dates between start and end"""
        cal = BC._trade_calendar if type == 'td' else BC._cds
        dates = cls.slice(cal, start, end)
        if until_today:
            dates = dates[dates <= cls.today()]
        if updated:
            dates = dates[dates <= cls.updated()]
        dates = dates[::step]
        if slice is not None:
            dates = cls.slice(dates, *slice)
        return dates

    @classmethod
    def range_segments(cls, start: DateTypeWithNone, end: DateTypeWithNone , type : Literal['td', 'cd'] = 'td' , 
                       step: int = 1 , until_today=True, updated=False,
                       slice: tuple[DateTypeWithNone, DateTypeWithNone] | None = None) -> list[tuple[int,int]]:
        """return the range of dates between start and end"""
        dates = cls.range(start, end, type, step=step , until_today=until_today, updated=updated, slice=slice)
        if len(dates) == 0:
            return []
        dt_starts = dates[::step]
        dt_ends = np.full_like(dt_starts , dates[-1])
        dt_ends[:-1] = dates[step-1::step]
        return [(s,e) for s,e in zip(dt_starts , dt_ends)]

    @classmethod
    def td_within(
        cls, start: DateTypeWithNone = None, end: DateTypeWithNone = None,
        step: int = 1, until_today=True, updated=False,
        slice: tuple[DateTypeWithNone, DateTypeWithNone] | None = None,
    ) -> np.ndarray[int, Any]:
        """All trading days within the interval; can be truncated to today, data update date, step and secondary 'slice'."""
        return cls.range(start, end, 'td', step=step, until_today=until_today, updated=updated, slice=slice)

    @classmethod
    def cd_within(
        cls, start: DateTypeWithNone = None, end: DateTypeWithNone = None,
        step: int = 1, until_today=True, updated=False,
        slice: tuple[DateTypeWithNone, DateTypeWithNone] | None = None,
    ) -> np.ndarray[int, Any]:
        """All natural days within the interval; can be truncated to today, data update date, step."""
        return cls.range(start, end, 'cd', step=step, until_today=until_today, updated=updated, slice=slice)

    @classmethod
    def diffs(cls, *args, type: Literal['td', 'cd'] = 'td'):
        """
        Calculate the difference between two arrays of dates.

        inputs *args options:
            1. start , end , source_dates: target_dates will be dates within start and end
            2. target_dates , source_dates.
        type : 'td' or 'cd'
        
        """
        assert len(args) in [2, 3], "Use tuple date_target must be a tuple of length 2 or 3"
        if len(args) == 2:
            target_dates, source_dates = args
        else:
            start, end, source_dates = args
            target_dates = cls.range(start, end , type) 
        if source_dates is None:
            source_dates = np.array([], dtype=int)
        return Dates(np.setdiff1d(target_dates, source_dates))

    @classmethod
    def td_filter(cls, dates: DatesType):
        """filter the trading dates in a date list."""
        cds = get_cds(dates)
        tds = cls.range(min(cds), max(cds), 'td')
        return np.intersect1d(tds, cds)

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
        cls , reference_date : DateType,
        period_num: int = 1, freq: Literal["d", "w", "m", "q", "y"] | str = "m", lag_num: int = 0,
    ):
        """Calculate the start and end date of a trading date interval based on the approximate trading date length (d/w/m/q/y) from the reference date."""
        pdays = {"d": 1, "w": 7, "m": 21, "q": 63, "y": 252}[freq]
        start = cls.td(reference_date , - pdays * (period_num + lag_num) + 1)
        end = cls.td(reference_date , - pdays * lag_num)
        return start, end

    @classmethod
    def cd_start_end(
        cls, reference_date : DateType,
        period_num: int = 1, freq: Literal["d", "w", "m", "q", "y"] | str = "m", lag_num: int = 0,
    ):
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
    def as_trade_date(date: DateType):
        """Convert a natural date to a trading date."""
        return TradeDate(date)

    @staticmethod
    def is_trade_date(date: DateType):
        """Whether the date is a trading day; 'int(TradeDate)' is 'td', participate in the judgment."""
        return True if isinstance(date, TradeDate) else BC.is_trade_cd(date)

    @staticmethod
    def trade_dates():
        """All trading dates 'YYYYMMDD' in ascending order (copy)."""
        return BC._trade_calendar.copy()

    @classmethod
    def slice(cls, dates: DatesType, start: DateTypeWithNone = None, end: DateTypeWithNone = None, year: int | None = None) -> np.ndarray:
        """Filter the date sequence based on 'start', 'end' and 'year'."""
        dates = get_cds(dates)
        dates = dates[(dates >= cls.as_start_date(start)) & (dates <= cls.as_end_date(end))]
        if year is not None:
            dates = dates[(dates // 10000) == year]
        return dates.astype(int)

    @staticmethod
    def reformat(date: Any, old_fmt = "%Y%m%d", new_fmt : str | None = "%Y%m%d"):
        """Convert the date string format; return the original string if the two formats are the same."""
        if old_fmt != new_fmt and new_fmt is not None:
            return datetime.strptime(str(date), old_fmt).strftime(new_fmt)
        else:
            return date

    @classmethod
    def year_end(cls, date: DateType):
        """The last day of the year (December 31) of the year the date is in ('YYYYMMDD')."""
        return get_td(date) // 10000 * 10000 + 1231

    @classmethod
    def year_start(cls, date: DateType):
        """The first day of the year (January 1) of the year the date is in ('YYYYMMDD')."""
        return get_td(date) // 10000 * 10000 + 101

    @classmethod
    def quarter_end(cls, date: DateType):
        """The last day of the quarter."""
        return cls.QE[cls.QE >= get_td(date)][0]

    @classmethod
    def quarter_start(cls, date: DateType):
        """The first day of the quarter."""
        td = get_td(date)
        return td // 10000 * 10000 + (td % 10000 // 100 / 3).__floor__() * 300 + 101

    @classmethod
    def month_end(cls, date: DateType):
        """The last day of the month."""
        return cls.ME[cls.ME >= get_td(date)][0]

    @classmethod
    def month_start(cls, date: DateType):
        """The first day of the month."""
        return get_td(date) // 100 * 100 + 1

    @classmethod
    def qe_trailing(cls, date: DateType, n_past=1, n_future=0, another_date: DateTypeWithNone = None, year_only=False):
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
        
        dates = cls.YE if year_only else cls.QE
        d0 = get_td(date)
        d1 = get_td(another_date) if another_date is not None else d0
        d0 , d1 = min(d0, d1), max(d0, d1)
        before = dates[dates <= d0][-n_past:]
        between = dates[(dates > d0) & (dates <= d1)]
        after = dates[dates > d1][:n_future]
        return np.concatenate([before, between, after])

    @classmethod
    def qe_within(cls, start : DateType, end : DateType, year_only : bool = False):
        """The set of quarter ends (or year ends) within the interval. optionally can only include year ends."""
        if year_only:
            return cls.YE[(cls.YE >= get_td(start)) & (cls.YE <= get_td(end))]
        return cls.QE[(cls.QE >= get_td(start)) & (cls.QE <= get_td(end))]

    @classmethod
    def qe_interpolate(cls, incomplete_qtr_ends: DatesType):
        """Expand the minimum and maximum range of several quarter ends to a complete quarter end sequence."""
        qes = get_cds(incomplete_qtr_ends)
        if len(qes) == 0:
            return qes
        return cls.qe_within(min(qes), max(qes))

    @classmethod
    def check_rollback_date(cls, rollback_date: DateTypeWithNone, max_rollback_days: int = 10):
        """Assert that 'rollback_date' is not earlier than max_rollback_days days before the updated date."""
        if rollback_date is None:
            return
        earliest_rollback_date = cls.td(cls.updated(), -max_rollback_days)
        assert get_td(rollback_date) >= earliest_rollback_date, (
            f"rollback_date {rollback_date} is too early, must be at least {earliest_rollback_date}"
        )


class Dates(np.ndarray[int, Any]):
    """
    An integer date array view.
    inputs:
        0. empty input:
            - empty dates[]
        1. only 1 input arg:
            - None : length 0 array
            - date : int / TradeDate / string of digits , length 1 array
            - dates : list / np.ndarray / pd.Series / Dates2 , directly converted to np.ndarray
        2. exact 2 inputs: start , end
        3. exact 3 inputs: dates , start , end
            - will use start and end to slice the dates
    """

    def __new__(cls, *args: Any, info=None):
        if len(args) == 0:
            dates = []
        elif len(args) == 1:
            if args[0] is None:
                dates = []
            elif isinstance(args[0], (int, TradeDate, str)):
                dates = [int(args[0])]
            else:
                dates = np.array(args[0], dtype=int)
        elif len(args) == 2:
            start, end = args
            dates = CALENDAR.range(start, end, 'td')
        elif len(args) == 3:
            dates, start, end = args
            dates = CALENDAR.slice(dates, start, end)
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
        obj = np.asarray(dates, dtype=int).view(cls)
        obj.info = info
        return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.format_str})"

    def __str__(self) -> str:
        return self.format_str()

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    @property
    def empty(self) -> bool:
        """no dates in the array"""
        return len(self) == 0

    def diffs(self, *others: Any) -> "Dates":
        """The difference set of dates."""
        return Dates(np.setdiff1d(self, Dates(*others)), info=self.info)

    def format_str(self) -> str:
        """short text description of empty, single or multiple days."""
        if self.empty:
            return "Empty Dates"
        if len(self) > 1:
            return f"{self.min()}~{self.max()}({len(self)}days)"
        return str(self[0])
