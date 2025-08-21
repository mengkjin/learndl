from pathlib import Path
from typing import Generator

from src.basic import CALENDAR , Logger
from src.func.dynamic_import import dynamic_members

from src.data.download.tushare.basic import TushareFetcher , TSBackUpDataTransform

class TushareDataDownloader:
    @classmethod
    def get_fetchers(cls) -> Generator[tuple[str , TushareFetcher] , None , None]:
        task_path = Path(__file__).parent
        for name , fetcher in dynamic_members(task_path , subclass_of=TushareFetcher):
            yield name , fetcher()

    @classmethod
    def update(cls):
        TSBackUpDataTransform.clear()
        for name , fetcher in cls.get_fetchers():
            try:
                fetcher.update()
            except Exception as e:
                Logger.error(f'{name} failed: {e}')
        TSBackUpDataTransform.update()

    @classmethod
    def update_rollback(cls , rollback_date : int):
        CALENDAR.check_rollback_date(rollback_date)
        TSBackUpDataTransform.rollback(rollback_date)
        for name , fetcher in cls.get_fetchers():
            try:
                fetcher.update_rollback(rollback_date)
            except Exception as e:
                Logger.error(f'{name} failed: {e}')
        TSBackUpDataTransform.update()