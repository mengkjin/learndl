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