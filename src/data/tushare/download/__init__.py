import inspect

from . import daily , fina , index , info
from ..basic import TushareFetecher

class TushareDownloader:
    DOWNLOAD_MODULES = [info , index , daily , fina]
    def __init__(self) -> None:
        pass

    @classmethod
    def proceed(cls):
        for module in cls.DOWNLOAD_MODULES:
            for name , task_cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(task_cls , TushareFetecher) and not task_cls.__abstractmethods__:
                    try:
                        task_cls().update()
                    except Exception as e:
                        print(f'{name} failed: {e}')
                        continue