import inspect

from .basic import TushareFetecher
from .download import daily , fina , index , info
from .model import TuShareCNE5_Calculator

def proceed():
    tushare_download()
    tushare_model()

def tushare_download():
    module_list = [info , index , daily , fina]
    for module in module_list:
        for name , task_cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(task_cls , TushareFetecher) or task_cls.__abstractmethods__:
                continue
            try:
                task_cls().update()
            except Exception as e:
                print(f'{name} failed: {e}')
                continue

def tushare_model():
    task_cne5 = TuShareCNE5_Calculator()
    task_cne5.Update('exposure')
    task_cne5.Update('risk')
