"""
Top-level orchestrator for all Tushare data fetchers.

``TushareDataDownloader`` iterates over all registered ``TushareFetcher``
subclasses (discovered via dynamic module import in ``TushareFetcher.load_tasks()``)
and calls their ``update()`` or ``rollback()`` methods.

``TSBackUpDataTransform.clear/update/rollback`` manages the manually-downloaded
CSV backup data that supplements the live Tushare pipeline.
"""
from typing import Generator , Type

from src.proj import Logger
from src.data.download.tushare.basic import TushareFetcher , TSBackUpDataTransform

class TushareDataDownloader:
    """Orchestrate incremental updates for all registered Tushare fetchers."""
    @classmethod
    def iter_fetchers(cls) -> Generator[Type[TushareFetcher] , None , None]:
        """iterate over all tushare fetchers"""
        TushareFetcher.load_tasks()
        for fetcher in TushareFetcher.registry.values():
            yield fetcher

    @classmethod
    def update(cls):
        """update all tushare fetchers"""
        Logger.note(f'Download: {cls.__name__} since last update!')
        TSBackUpDataTransform.clear()
        for fetcher in cls.iter_fetchers():
            fetcher.update()
        TSBackUpDataTransform.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        """update all tushare fetchers with rollback date"""
        Logger.note(f'Download: {cls.__name__} rollback from {rollback_date}!')
        TSBackUpDataTransform.rollback(rollback_date)
        for fetcher in cls.iter_fetchers():
            fetcher.rollback(rollback_date)
        TSBackUpDataTransform.update()