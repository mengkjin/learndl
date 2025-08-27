from pathlib import Path
from typing import Generator

from src.func.dynamic_import import dynamic_members
from src.data.download.tushare.basic import TushareFetcher , TSBackUpDataTransform

class TushareDataDownloader:
    @classmethod
    def get_fetchers(cls) -> Generator[TushareFetcher , None , None]:
        task_path = Path(__file__).parent
        for _ , fetcher in dynamic_members(task_path , subclass_of=TushareFetcher):
            yield fetcher()

    @classmethod
    def update(cls):
        TSBackUpDataTransform.clear()
        for fetcher in cls.get_fetchers():
            fetcher.update()
        TSBackUpDataTransform.update()

    @classmethod
    def update_rollback(cls , rollback_date : int):
        TSBackUpDataTransform.rollback(rollback_date)
        for fetcher in cls.get_fetchers():
            fetcher.update_rollback(rollback_date)
        TSBackUpDataTransform.update()