"""Trading calendar: ``TradeDate``, ``CALENDAR``, ``Dates``, and ``BJTZ``.

Implementation uses ndarray indexes and ``pandas.Index.get_indexer`` instead of heavy
``DataFrame`` indexing on hot paths. Import ``TradeDate`` only from this package so
``isinstance`` checks stay consistent across the codebase.
"""

from .basic import BJ_TZ , TradeDate , intDate , intDateNone , intDates
from .cal import CALENDAR
from .dates import Dates

__all__ = ["CALENDAR", "Dates" , "BJ_TZ" , "TradeDate"]
