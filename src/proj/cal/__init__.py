"""Trading calendar: ``TradeDate``, ``CALENDAR``, ``Dates``, and ``BJTZ``.

Implementation uses ndarray indexes and ``pandas.Index.get_indexer`` instead of heavy
``DataFrame`` indexing on hot paths. Import ``TradeDate`` only from this package so
``isinstance`` checks stay consistent across the codebase.
"""

from .cal import CALENDAR
from .dates import Dates

__all__ = ["CALENDAR", "Dates"]
