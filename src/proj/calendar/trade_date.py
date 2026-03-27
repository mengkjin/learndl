"""
Trading date transformation.
"""
import numpy as np
from typing import Any
from .basic import BC

class TradeDate:
    """'TradeDate' represents a date in the trading date perspective. input date is in 'YYYYMMDD' format."""
    def __new__(cls, date: int | Any, *args, **kwargs):
        if isinstance(date, TradeDate):
            return date
        return super().__new__(cls)

    def __init__(self, date: int | Any, force_trade_date=False):
        """
        Args:
            date: natural date or already a 'TradeDate' (the latter will not be re-initialized).
            force_trade_date: if True, skip the calendar mapping, 'td = cd'.
        """
        if not isinstance(date, TradeDate):
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
