import inspect

from .basic.connect import pro
from .download import TushareFetcher , daily , fina , index , info , analyst

class TushareDataDownloader:
    @classmethod
    def update(cls):
        module_list = [info , index , daily , fina , analyst]
        for module in module_list:
            for name , task_cls in inspect.getmembers(module, inspect.isclass):
                if not issubclass(task_cls , TushareFetcher) or task_cls.__abstractmethods__:
                    continue
                try:
                    task_cls().update()
                except Exception as e:
                    print(f'{name} failed: {e}')
                    continue