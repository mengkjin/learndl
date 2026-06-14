"""
Basic calendar tools: build and query the underlying calendar table.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from functools import cached_property
from typing import Any, TypeAlias, Union, Iterable
from zoneinfo import ZoneInfo

from src.proj.core import SingletonMeta , lit , as_int_array
from src.proj.env import MACHINE , PATH

__all__ = ['BJ_TZ', 'LOCAL_TZ', 'BasicCalendar', 'intDate', 'intDateNone', 'intDates']

BJ_TZ = ZoneInfo("Asia/Shanghai")
LOCAL_TZ = MACHINE.timezone

def get_cd(date: intDate) -> int:
    """Natural calendar day ``YYYYMMDD`` as int; for ``TradeDate``, returns ``.cd``."""
    return int(date.cd if isinstance(date, TradeDate) else date)

def get_td(date: intDate) -> int:
    """Trading-aligned day ``YYYYMMDD`` as int; for ``TradeDate``, returns ``.td``."""
    return int(date.td if isinstance(date, TradeDate) else date)

class BasicCalendar(metaclass=SingletonMeta):
    """The full calendar built from information_ts/calendar and configuration; keep 'full'/'cal'/'trd' DataFrame for reference."""

    def __init__(self) -> None:
        """Load and merge the calendar, build 'full'/'cal'/'trd' DataFrame and accelerate use ndarray and index."""
        self._loaded = False

    def ensure_data(self) -> None:
        if self._loaded:
            return
        calendar = pd.read_feather(PATH.db.joinpath("DB_information_ts" , "calendar.feather")).loc[:, ["calendar", "trade"]]
        reserved = pd.DataFrame(MACHINE.config.get('constant/data/calendar'))
        if not reserved.empty:
            calendar = pd.concat([calendar, reserved.loc[:, ["calendar", "trade"]]])
        calendar = calendar.drop_duplicates(subset="calendar", keep="first").sort_values("calendar")

        trd = calendar.query("trade == 1").reset_index(drop=True)
        trd["td"] = trd["calendar"]
        trd["pre"] = trd["calendar"].shift(1, fill_value=-1)
        calendar = calendar.merge(trd.drop(columns="trade"), on="calendar", how="left").ffill()
        calendar["cd_index"] = np.arange(len(calendar))
        calendar["td_index"] = calendar["trade"].cumsum() - 1
        calendar["td_forward_index"] = calendar["td_index"] + 1 - calendar["trade"]
        calendar["td_forward"] = trd.iloc[
            calendar["td_forward_index"].clip(None, len(trd) - 1).to_numpy(int)
        ]["calendar"].values
        calendar = calendar.astype(int).set_index("calendar")
        cal_cal = calendar.reset_index().set_index("cd_index")
        cal_trd = calendar[calendar["trade"] == 1].reset_index().set_index("td_index")

        self.full = calendar
        self.cal = cal_cal
        self.trd = cal_trd

        self._loaded = True

    @cached_property
    def full(self) -> pd.DataFrame:
        self.ensure_data()
        return self.full

    @cached_property
    def _cds(self) -> np.ndarray:
        self.ensure_data()
        return self.full.index.to_numpy(dtype=np.int64)

    @cached_property
    def _cd_to_pos(self) -> dict[int, int]:
        return {int(c): i for i, c in enumerate(self._cds)}

    @cached_property
    def _trade(self) -> np.ndarray:
        return self.full["trade"].to_numpy(dtype=bool)

    @cached_property
    def _td_col(self) -> np.ndarray:
        return self.full["td"].to_numpy(dtype=np.int64)

    @cached_property
    def _cd_index(self) -> np.ndarray:
        return self.full["cd_index"].to_numpy(dtype=np.int64)

    @cached_property
    def _td_index(self) -> np.ndarray:
        return self.full["td_index"].to_numpy(dtype=np.int64)

    @cached_property
    def _td_forward(self) -> np.ndarray:
        return self.full["td_forward"].to_numpy(dtype=np.int64)

    @cached_property
    def _trade_calendar(self) -> np.ndarray:
        return self.full.index.to_numpy(dtype=np.int64)[self._trade].astype(np.int64, copy=False)

    @cached_property
    def _trade_td(self) -> np.ndarray:
        return self.full["td"].to_numpy(dtype=np.int64)[self._trade]

    @cached_property
    def min_date(self) -> int:
        return int(self._cds.min())
    @cached_property
    def max_date(self) -> int:
        return int(self._cds.max())
    @cached_property
    def n_cal(self) -> int:
        return len(self._cds)
    @cached_property
    def n_td(self) -> int:
        return int(self._trade.sum())
    @cached_property
    def max_td_index(self) -> int:
        return int(self._td_index.max())
    @cached_property
    def _cd_pd_index(self):
        from pandas import Index
        return Index(self._cds)

    def pos_cd(self, cd: int) -> int:
        """The row number of the natural date 'YYYYMMDD' in the internal sorted table; raise 'KeyError' if not exists."""
        return self._cd_to_pos[cd]

    def pos_cd_array(self, cd_arr: np.ndarray) -> np.ndarray:
        """The row numbers of the natural dates 'YYYYMMDD'; raise 'KeyError' if any key not in the calendar."""
        cd_arr = np.asarray(cd_arr, dtype=np.int64)
        pos = self._cd_pd_index.get_indexer(cd_arr.tolist())
        if (pos < 0).any():
            raise KeyError("date not in calendar")
        return pos.astype(np.int64, copy=False)

    def td_for_cd(self, cd: int) -> int:
        """The trading date 'YYYYMMDD' corresponding to the natural date (non-trading day is forward filled result)."""
        return int(self._td_col[self.pos_cd(cd)])

    def td_forward_for_cd(self, cd: int) -> int:
        """The next trading date 'YYYYMMDD' corresponding to the natural date (consistent with the 'td_forward' column in the old table)."""
        return int(self._td_forward[self.pos_cd(cd)])

    def td_index_for_cd(self, cd: int) -> int:
        """The trading date index of the natural date in the full calendar (0..n_td-1, consistent with the 'td_index' column in the old table)."""
        return int(self._td_index[self.pos_cd(cd)])

    def cd_index_for_cd(self, cd: int) -> int:
        """The index of the natural date in the continuous natural date sequence."""
        return int(self._cd_index[self.pos_cd(cd)])

    def is_trade_cd(self, cd: int) -> bool:
        """Whether the natural date 'cd' is a trading day."""
        return self._trade[self.pos_cd(cd)]

    def trade_date_by_td_index(self, td_index: int | np.ndarray) -> int | np.ndarray:
        """Take the 'td' column by the trading date index"""
        if isinstance(td_index, np.ndarray):
            return self._trade_td[td_index]
        return int(self._trade_td[int(td_index)])

    def trade_calendar_by_td_index(self, td_index: int | np.ndarray) -> int | np.ndarray:
        """Take the natural date 'YYYYMMDD' by the trading date index"""
        if isinstance(td_index, np.ndarray):
            return self._trade_calendar[td_index]
        return int(self._trade_calendar[int(td_index)])

    def calendar_by_cd_index(self, cd_index: int | np.ndarray) -> int | np.ndarray:
        """Take the natural date 'YYYYMMDD' by the natural date index"""
        if isinstance(cd_index, np.ndarray):
            cd_index = np.clip(cd_index, 0, self.n_cal - 1)
            return self._cds[cd_index]
        i = int(cd_index)
        i = max(0, min(i, self.n_cal - 1))
        return int(self._cds[i])

    def trailing_np(self, date: intDate, n: int, type: lit.intDateType = 'td') -> np.ndarray:
        """The last 'n' trading days before 'date' (sorted slice)."""
        date = get_td(date)
        cal = self._trade_calendar if type == 'td' else self._cds
        last = int(np.searchsorted(cal, date, side="right")) - 1
        if last < 0:
            return np.array([], dtype=np.int64)
        start = max(0, last - n + 1)
        return cal[start : last + 1].astype(np.int64, copy=False)

    def offset_np(self, dates: intDates, offset: int = 0, type: lit.intDateType = 'td' , backward=True) -> np.ndarray:
        """Convert multiple natural dates to trading dates (or 'td_forward'); optionally offset by 'offset' steps."""
        dates = as_int_array(dates)
        if len(dates) == 0:
            return dates
        if type == 'td':
            pos = BC.pos_cd_array(dates)
            td_arr = (BC._td_col if backward else BC._td_forward)[pos]
            td_arr = np.asarray(td_arr)
            if offset != 0:
                pos_td = BC.pos_cd_array(td_arr)
                d_index = BC._td_index[pos_td] + offset
                d_index = np.maximum(np.minimum(d_index, BC.n_td - 1), 0)
                td_arr = BC.trade_calendar_by_td_index(d_index)
                td_arr = np.asarray(td_arr)
            return td_arr.astype(np.int64, copy=False)
        else:
            cd_arr = as_int_array(dates)
            if offset != 0:
                cd_arr = np.minimum(cd_arr, BC.max_date)
                d_index = BC._cd_index[BC.pos_cd_array(cd_arr)] + offset
                d_index = np.maximum(np.minimum(d_index, BC.n_cal - 1), 0)
                cd_arr = np.asarray(BC.calendar_by_cd_index(d_index))
            return cd_arr.astype(np.int64, copy=False)

    def diff_days(self, date1 : intDate, date2 : intDate , type : lit.intDateType = 'td') -> int | Any:
        """The difference in days between two dates."""
        try:
            i1 = self.pos_cd(get_cd(date1))
            i2 = self.pos_cd(get_cd(date2))
            if type == 'td':
                diff = int(self._td_index[i2] - self._td_index[i1])
            elif type == 'cd':
                diff = int(self._cd_index[i2] - self._cd_index[i1])
            else:
                raise ValueError(f"Invalid date type: {type}")
        except KeyError as e:
            raise ValueError(f"{date1} and {date2} are not in {type} calendar {e!s}")
        return diff

    def diff_days_array(self, date1_arr : intDates, date2_arr : intDates , type : lit.intDateType = 'td') -> int | Any:
        """The difference of two arrays of dates."""
        p1 = self.pos_cd_array(as_int_array(date1_arr))
        p2 = self.pos_cd_array(as_int_array(date2_arr))
        if type == 'td':
            return self._td_index[p1] - self._td_index[p2]
        elif type == 'cd':
            return self._cd_index[p1] - self._cd_index[p2]
        else:
            raise ValueError(f"Invalid date type: {type}")

    def range(
        self , start: intDateNone, end: intDateNone , 
        type : lit.intDateType = 'td' , step: int = 1
    ) -> np.ndarray[Any, np.dtype[np.int_]]:
        """return the range of dates between start and end"""
        dates = as_int_array(self._trade_calendar if type == 'td' else self._cds)
        dates = dates[(dates >= int(start or 0)) & (dates <= int(end or 99991231))]
        dates = dates[::step]
        return dates

class TradeDate:
    """'TradeDate' represents a date in the trading date perspective. input date is in 'YYYYMMDD' format."""
    def __init__(self, date: int | TradeDate | Any, force_trade_date=False):
        """
        Args:
            date: natural date or already a 'TradeDate' (the latter will not be re-initialized).
            force_trade_date: if True, skip the calendar mapping, 'td = cd'.
        """
        if isinstance(date, TradeDate):
            self.cd = date.cd
            self.td = date.td
        else:
            self.cd = int(date)
            if force_trade_date or self.cd < BC.min_date or self.cd > BC.max_date:
                self.td: int = self.cd
            else:
                self.td = BC.td_for_cd(self.cd)

    def __repr__(self):
        return str(self.td)

    def __int__(self):
        """return the trading date 'td'."""
        return int(self.td)

    def __str__(self):
        return str(self.td)

    def __add__(self, n: int):
        """move forward 'n' trading days (can be negative)."""
        return self.offset(n)

    def __sub__(self, n: int):
        """move backward 'n' trading days."""
        return self.offset(-n)

    def __lt__(self, other):
        return int(self) < int(other)

    def __le__(self, other):
        return int(self) <= int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __ge__(self, other):
        return int(self) >= int(other)

    def __eq__(self, other):
        return int(self) == int(other)

    def as_int(self):
        """return the trading date 'td' as an integer."""
        return int(self)

    def offset(self, n: int):
        """move 'n' trading days (can be negative), clip the index to the valid range if out of bounds."""
        return self._cls_offset(self, n)

    @classmethod
    def _cls_offset(cls, td0, n: int):
        td0 = cls(td0)
        if n == 0:
            return td0
        elif td0 < BC.min_date or td0 > BC.max_date:
            return td0
        assert isinstance(n, (int, np.integer)), f"n must be a integer, got {type(n)}"
        d_index = BC.td_index_for_cd(td0.td) + n
        d_index = np.maximum(np.minimum(d_index, BC.max_td_index), 0)
        new_date = BC.trade_date_by_td_index(d_index)
        return cls(new_date)


BC = BasicCalendar()
intDate : TypeAlias = Union[int , TradeDate]
intDateNone : TypeAlias = Union[int, TradeDate, None]
intDates : TypeAlias = Union[intDate , Iterable[intDate] , np.ndarray , pd.Series]