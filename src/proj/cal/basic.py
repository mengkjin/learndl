"""
Basic calendar tools: build and query the underlying calendar table.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from functools import cached_property
from zoneinfo import ZoneInfo

from src.proj.core import SingletonMeta
from src.proj.env import MACHINE
import src.proj.db as DB

#: Shanghai time zone, used by CALENDAR , TradeDate etc.
BJ_TZ = ZoneInfo("Asia/Shanghai")
LOCAL_TZ = MACHINE.timezone

class BasicCalendar(metaclass=SingletonMeta):
    """The full calendar built from information_ts/calendar and configuration; keep 'full'/'cal'/'trd' DataFrame for reference."""

    def __init__(self) -> None:
        """Load and merge the calendar, build 'full'/'cal'/'trd' DataFrame and accelerate use ndarray and index."""
        self._loaded = False

    def ensure_data(self) -> None:
        if self._loaded:
            return
        calendar = DB.load("information_ts", "calendar", missing_ok=False).loc[:, ["calendar", "trade"]]
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

    def td_trailing_np(self, date: int, n: int) -> np.ndarray:
        """The last 'n' trading days before 'date' (sorted slice; 'CALENDAR.td_trailing' will still 'np.sort' to align with the old interface)."""
        t = self._trade_calendar
        last = int(np.searchsorted(t, date, side="right")) - 1
        if last < 0:
            return np.array([], dtype=np.int64)
        start = max(0, last - n + 1)
        return t[start : last + 1].astype(np.int64, copy=False)

    def cd_trailing_np(self, date: int, n: int) -> np.ndarray:
        """The last 'n' natural days before 'date' (sorted slice)."""
        c = self._cds
        last = int(np.searchsorted(c, date, side="right")) - 1
        if last < 0:
            return np.array([], dtype=np.int64)
        start = max(0, last - n + 1)
        return c[start : last + 1].astype(np.int64, copy=False)

BC = BasicCalendar()