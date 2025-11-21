from typing import Generator , Type

from src.data.download.tushare.basic import TushareFetcher , TSBackUpDataTransform

class TushareDataDownloader:
    @classmethod
    def iter_fetchers(cls) -> Generator[Type[TushareFetcher] , None , None]:
        """iterate over all tushare fetchers"""
        TushareFetcher.load_tasks()
        for fetcher in TushareFetcher.registry.values():
            yield fetcher

    @classmethod
    def update(cls):
        """update all tushare fetchers"""
        TSBackUpDataTransform.clear()
        for fetcher in cls.iter_fetchers():
            fetcher.update()
        TSBackUpDataTransform.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        """update all tushare fetchers with rollback date"""
        TSBackUpDataTransform.rollback(rollback_date)
        for fetcher in cls.iter_fetchers():
            fetcher.rollback(rollback_date)
        TSBackUpDataTransform.update()