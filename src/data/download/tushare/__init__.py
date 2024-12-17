from pathlib import Path

from src.func.dynamic_import import dynamic_members , true_subclass

from .basic import pro , TushareFetcher
from . import task

class TushareDataDownloader:
    @classmethod
    def update(cls):
        for name , fetcher in dynamic_members(getattr(task , '__path__')[0] , lambda x: true_subclass(x , TushareFetcher)):
            assert issubclass(fetcher , TushareFetcher) , f'{name} is not a valid TushareFetcher'
            try:
                fetcher().update()
            except Exception as e:
                print(f'{name} failed: {e}')
                continue