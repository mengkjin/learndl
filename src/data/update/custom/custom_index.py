"""
Custom benchmark index framework and updater.

Provides an abstract ``CustomIndex`` base class for building custom equal- or
cap-weighted portfolio benchmarks, plus a ``CustomIndexUpdater`` that incrementally
computes and stores daily portfolio weights.

Currently registered indices
----------------------------
- ``MicroCap_400``: equal-weight portfolio of up to 400 smallest non-ST A-share stocks.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from abc import ABCMeta , abstractmethod
from functools import cached_property
from typing import Any

from src.proj import CALENDAR , DB , Load , Save , Base , Dates
from src.data.loader.data_vendor import DATAVENDOR
from src.data.update.custom.basic import BasicCustomUpdater

__all__ = ['CustomIndex' , 'CustomIndexUpdater' , 'MicroCap_400']

START_DATE = 20050101
DB_SRC = 'index_daily_custom'

class CustomIndexMeta(ABCMeta):
    """Metaclass that auto-registers concrete ``CustomIndex`` subclasses."""
    registry : dict[str , type[CustomIndex] | Any] = {}
    def __new__(cls , name , bases , dct):
        new_cls = super().__new__(cls , name , bases , dct)
        assert name not in cls.registry or cls.registry[name].__module__ == new_cls.__module__ , \
            f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
        if not new_cls.__abstractmethods__:
            cls.registry[name] = new_cls
        return new_cls

class CustomIndexName:
    """Descriptor that returns the lowercased class name as the index identifier."""
    def __get__(self , instance , owner) -> str:
        """Return ``owner.__name__.lower()`` as the index name."""
        return owner.__name__.lower()

class CustomIndex(metaclass=CustomIndexMeta):
    """
    Abstract base class for custom benchmark indices.

    Subclasses must implement:
    - ``rebalance_dates()`` → array of rebalance dates
    - ``rebalance_portfolio(date)`` → DataFrame of weights at ``date``

    ``index_portfolio(date)`` lazily evolves the portfolio forward from the most
    recent rebalance date, using price returns via ``Port.evolve_to_date``.
    """
    START_DATE : int = START_DATE
    index_name = CustomIndexName()

    @abstractmethod
    def rebalance_dates(self) -> Dates:
        """get rebalance dates"""

    @abstractmethod
    def rebalance_portfolio(self , date : int) -> pd.DataFrame:
        """get index portfolio"""

    @cached_property
    def reb_dates(self) -> Dates:
        """Cached rebalance date array (computed on first access)."""
        return self.rebalance_dates()

    @property
    def target_path(self):
        """Filesystem path for this index's daily weight storage."""
        return DB.path(DB_SRC , self.index_name)

    @property
    def current_portfolio(self):
        """The most recently evolved portfolio object, or None if not yet computed."""
        from src.res.factor.util.classes.port import Port
        if hasattr(self , '_current_portfolio'):
            assert isinstance(self._current_portfolio , Port) , f'current_portfolio is not a Port instance: {type(self._current_portfolio)}'
            return self._current_portfolio
        return None

    @current_portfolio.setter
    def current_portfolio(self , port):
        self._current_portfolio = port

    def index_portfolio(self , date : int):
        """get index portfolio"""
        from src.res.factor.util.classes.port import Port
        if self.current_portfolio is not None and self.current_portfolio.date == date:
            return self.current_portfolio
        reb_dates = self.reb_dates
        prev_reb_date = 99991231 if (not reb_dates or reb_dates.min > date) else reb_dates.slice(end = date).max
        if prev_reb_date > date:
            port = Port(pd.DataFrame() , date , self.index_name)
        elif prev_reb_date == date:
            port = Port(self.rebalance_portfolio(date) , date , self.index_name)
        elif self.current_portfolio is not None and self.current_portfolio.date >= prev_reb_date:
            port = self.current_portfolio.evolve_to_date(date)
        else:
            prev_reb_port = Port(self.rebalance_portfolio(prev_reb_date) , prev_reb_date , self.index_name)
            port = prev_reb_port.evolve_to_date(date)
        self.current_portfolio = port
        return self.current_portfolio

    def index_return(self , date : int) -> float:
        """get index return"""
        day = DB.load('trade_ts', 'day', int(date), vb_level='never')
        if day is None or day.empty or not {'secid', 'close', 'adjfactor'}.issubset(day.columns):
            raise ValueError(f'Cannot calculate {self.index_name} at {date}: trade_ts/day is missing, empty or lacks required columns')
        prev_date = CALENDAR.td(date , -1).td
        prev_port = self.index_portfolio(prev_date)
        if prev_port.emtpy:
            return 0.
        return prev_port.fut_ret(date)

    def update_dates(self , dates : Base.alias.DateType , indent : int = 1 , vb_level : Base.lit.VerbosityLevel = 1) -> None:
        """update index return for given dates"""
        dates = Dates(dates)
        if dates.empty:
            return
        dates = dates.slice(self.START_DATE , CALENDAR.updated())
        pct_chg = np.array([self.index_return(date) for date in dates]) * 100
        df = pd.DataFrame({'trade_date' : dates.dates , 'pct_chg' : pct_chg})
        old_df = Load.df(self.target_path)
        if not old_df.empty:
            df = pd.concat([old_df.query('trade_date not in @df.trade_date') , df]).sort_values('trade_date').reset_index(drop=True)
        Save.df(df , self.target_path , indent = indent , vb_level = vb_level)

    def stored_dates(self) -> Dates:
        """get stored dates"""
        old_df = Load.df(self.target_path)
        if old_df.empty:
            return Dates()
        return Dates(old_df['trade_date'].to_numpy(int))

    def target_dates(self , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs) -> Dates:
        start = max(start or self.START_DATE , self.START_DATE)
        end = end or CALENDAR.update_to()
        stored_dates = Dates() if overwrite else self.stored_dates()
        target_dates = Dates(start , end).diff(stored_dates)
        return target_dates

    @classmethod
    def iter_custom_indices(cls , names : list[str] | None = None):
        name_set = set(names) if names is not None else None
        for name , custom_index_cls in cls.registry.items():
            if name_set is not None and name not in name_set:
                continue
            yield custom_index_cls()

class CustomIndexUpdater(BasicCustomUpdater):
    START_DATE = START_DATE

    @classmethod
    def proceed_update(
        cls , start : int | None = None , end : int | None = None , overwrite : bool = False ,
        index_names : list[str] | None = None , **kwargs
    ) -> Base.UpdateFlag:
        total_dates = []
        custom_indices = list(CustomIndex.iter_custom_indices(index_names))
        for custom_index in custom_indices:
            target_dates = custom_index.target_dates(start = start , end = end , overwrite = overwrite)
            if len(target_dates) == 0:
                cls.logger.skipping(f'{custom_index.index_name} is up to date' , idt = 2 , vb = 1)
                continue
            custom_index.update_dates(target_dates , indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2)
            total_dates.extend(target_dates.dates.tolist())
        if len(total_dates) == 0:
            return Base.UpdateFlag.SKIPPED
        else:
            return Base.UpdateFlag.SUCCESS

    def update_one(self , date : int):
        for custom_index in CustomIndex.iter_custom_indices():
            custom_index.update_dates([date] , indent = self.indent + 2 , vb_level = self.vb_level + 2)

class MicroCap_400(CustomIndex):
    def rebalance_dates(self) -> Dates:
        # START_DATE is the first return date, so its opening portfolio must
        # already exist at the preceding trading day's close.
        first_return = CALENDAR.td(self.START_DATE, backward=False).as_int()
        first_holdings = CALENDAR.td(first_return, -1).as_int()
        return Dates(first_holdings, CALENDAR.updated())

    def rebalance_portfolio(self , date : int) -> pd.DataFrame:
        prev_date = CALENDAR.td(date , -1)
        prev_val = DATAVENDOR.TRADE.get_val(prev_date)
        df = pd.DataFrame({'secid' : prev_val['secid'] , 'free_cap' : prev_val['free_share'] * prev_val['close']})
        pool = DATAVENDOR.INFO.get_desc(prev_date , exchange = ['SZSE', 'SSE'] , set_index=False)['secid']
        st_stocks = DATAVENDOR.INFO.get_st(prev_date)['secid']
        # Count the listed SSE/SZSE universe before excluding ST constituents.
        # Floor odd counts so the index never exceeds half of that universe.
        count = min(400, pool.nunique() // 2)
        trading = DATAVENDOR.TRADE.get_trd(date)
        if trading.empty or not {'secid', 'status', 'close'}.issubset(trading.columns):
            raise ValueError(f'Cannot build {self.index_name} holdings at {date}: missing trading-status quotes')
        active = trading.loc[(trading.status == 1) & np.isfinite(trading.close) & (trading.close > 0), 'secid']
        df = df[df.secid.isin(active)]
        df = df[df['secid'].isin(pool) & ~df['secid'].isin(st_stocks)].sort_values('free_cap' , ascending=True).\
            head(count).reset_index(drop=True).drop(columns=['free_cap'])
        if df.empty:
            raise ValueError(f'Cannot build {self.index_name} holdings at {date}: no eligible constituents')
        df['weight'] = 1 / len(df)
        return df
