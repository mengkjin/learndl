from typing import Any
from . import control , display , model
            
class ModelHook:
    def __init__(self , ptimer = None) -> None:
        self._tm = ptimer if ptimer is not None else self.EmptyTM
    def update_timer(self , ptimer): self.tm = ptimer
    def hook(self , func):
        def wrapper(*args , **kwargs):
            with self._tm(func.__name__):
                func(*args , **kwargs)
                self._hook_call(args[0] , func.__name__)
        return wrapper
    def _hook_call(self , obj , hook_name) -> None:
        [getattr(cb , hook_name)(obj) for cb in obj.callbacks if hasattr(cb , hook_name)]

    class EmptyTM:
        def __init__(self , *args): pass
        def __enter__(self): pass
        def __exit__(self , *args): pass