from src.basic import CALENDAR
from src.func.dynamic_import import dynamic_members

from .basic import pro , TushareFetcher , TSBackUpDataTransform
from . import task

class TushareDataDownloader:
    @classmethod
    def update(cls):
        TSBackUpDataTransform.clear()
        for name , fetcher in dynamic_members(getattr(task , '__path__')[0] , subclass_of=TushareFetcher):
            try:
                fet : TushareFetcher = fetcher()
                fet.update()
            except Exception as e:
                print(f'{name} failed: {e}')
                continue
        TSBackUpDataTransform.update()

    @classmethod
    def update_rollback(cls , rollback_date : int):
        assert rollback_date >= CALENDAR.earliest_rollback_date() , \
            f'rollback_date {rollback_date} is too early, must be at least {CALENDAR.earliest_rollback_date()}'
        TSBackUpDataTransform.rollback(rollback_date)
        for name , fetcher in dynamic_members(getattr(task , '__path__')[0] , subclass_of=TushareFetcher):
            try:
                fet : TushareFetcher = fetcher()
                fet.update_rollback(rollback_date)
            except Exception as e:
                print(f'{name} failed: {e}')
                continue