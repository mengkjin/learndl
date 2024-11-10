import inspect

from . import analyst, daily , fina , index , info
from ..basic import TushareFetcher

class TushareDownloader:
    '''download data from tushare'''
    DOWNLOAD_MODULES = [info , index , daily , fina , analyst] # order matters
    def __init__(self) -> None:
        pass

    @classmethod
    def proceed(cls):
        for module in cls.DOWNLOAD_MODULES:
            for name , task_cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(task_cls , TushareFetcher) and not task_cls.__abstractmethods__:
                    try:
                        task_cls().update()
                    except Exception as e:
                        print(f'{name} failed: {e}')
                        continue