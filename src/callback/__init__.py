from inspect import currentframe
from typing import Any
from . import control , display , model
from ..util.classes import BaseCallBack , WithCallBack
class ModelHook:
    def __init__(self , ptimer = None) -> None:
        self._tm = ptimer if ptimer is not None else self.EmptyTM
    def update_timer(self , ptimer): self.tm = ptimer
    def hook(self , func):
        def wrapper(*args , **kwargs):
            with self._tm(func.__name__):
                func(*args , **kwargs)
                [cb(func.__name__ , args[0]) for cb in args[0].callbacks]
        return wrapper

    class EmptyTM:
        def __init__(self , *args): pass
        def __enter__(self): pass
        def __exit__(self , *args): pass

class CallBackManager:
    def __init__(self , *callbacks):        
        self.callbacks = callbacks
        self.with_cbs : list[WithCallBack] = []
        self.base_cbs : list[BaseCallBack] = []
        for cb in self.callbacks:
            if self.is_withcallback(cb):
                self.with_cbs.append(cb)
            else:
                self.base_cbs.append(cb)

    @staticmethod
    def is_withcallback(cb): return issubclass(cb.__class__ , WithCallBack)

    def __enter__(self):
        hook_name = str(getattr(currentframe() , 'f_back').f_code.co_name)
        assert hook_name.startswith('on_') , hook_name
        self.hook_name = hook_name
        for cb in self.with_cbs: cb.__enter__()

    def __exit__(self , *args):
        for cb in self.base_cbs: cb(self.hook_name)
        for cb in self.with_cbs: cb.__exit__()
        for cb in self.with_cbs: cb(self.hook_name)
