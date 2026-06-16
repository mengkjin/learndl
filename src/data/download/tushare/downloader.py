"""
Top-level orchestrator for all Tushare data fetchers.

``TushareDataDownloader`` iterates over all registered ``TushareFetcher``
subclasses (discovered via dynamic module import in ``TushareFetcher.load_tasks()``)
and calls their ``update()`` or ``rollback()`` methods.

``TSBackUpDataTransform.clear/update/rollback`` manages the manually-downloaded
CSV backup data that supplements the live Tushare pipeline.
"""
from __future__ import annotations
from typing import Any
from collections.abc import Generator

from src.proj import Base
from src.data.download.tushare.basic import TushareFetcher , TSBackUpDataTransform

__all__ = ['TushareDataDownloader']

class TushareDataDownloader(Base.BasicUpdater):
    """Orchestrate incremental updates for all registered Tushare fetchers."""
    UPDATE_ALIAS = 'download'
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE , Base.UpdateType.ROLLBACK)
    @classmethod
    def iter_fetchers(cls) -> Generator[type[TushareFetcher] , None , None]:
        """iterate over all tushare fetchers"""
        TushareFetcher.load_tasks()
        for fetcher in TushareFetcher.registry.values():
            yield fetcher

    @classmethod
    def proceed_update(
        cls , update_type : Base.UpdateType , rollback_date : int | None = None , 
        indent : int = 0 , vb_level : Any = 1 , **kwargs
    ) -> Base.UpdateFlag:
        flags = Base.UpdateFlagList()
        rollback_date = rollback_date if update_type == Base.UpdateType.ROLLBACK else None
        TSBackUpDataTransform.clear(rollback_date = rollback_date)
        for fetcher in cls.iter_fetchers():
            flags += fetcher.update(rollback_date = rollback_date , indent = indent + 1 , vb_level = vb_level + 1)
        TSBackUpDataTransform.update()
        return flags.summarize()