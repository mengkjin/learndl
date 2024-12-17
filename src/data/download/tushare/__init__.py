from pathlib import Path

from src.func.dynamic_import import dynamic_members

from .basic import pro , TushareFetcher
from . import task

class TushareDataDownloader:
    @classmethod
    def update(cls):
        for name , fetcher in dynamic_members(getattr(task , '__path__')[0] , subclass_of=TushareFetcher):
            try:
                fetcher().update()
            except Exception as e:
                print(f'{name} failed: {e}')
                continue