from typing import Any , Optional

from . import base , algo , control , display
from .base import BasicCallBack , WithCallBack
from ..config import TrainConfig

_search_cb_mod = [control , display , algo]

class CallBackManager(WithCallBack):
    def __init__(self , model_module , *callbacks):
        super().__init__(model_module)     
        self.callbacks = []
        self.with_cbs : list[WithCallBack] = []
        self.base_cbs : list[BasicCallBack] = []
        [self.add_callback(cb) for cb in callbacks]

    def add_callback(self , cb):
        self.callbacks.append(cb)
        self.with_cbs.append(cb) if issubclass(cb.__class__ , WithCallBack) else self.base_cbs.append(cb)

    def at_enter(self , hook_name):
        for cb in self.with_cbs: cb.at_enter(hook_name)
    def at_exit(self, hook_name):
        for cb in self.base_cbs: cb(hook_name)
        for cb in self.with_cbs: cb.at_exit(hook_name)

    @classmethod
    def setup(cls , config : TrainConfig , model_module : Any):
        callbacks = [cls.__get_cb(cb_name , param , config , model_module) for cb_name , param in config.callbacks.items()]
        return cls(model_module , *callbacks)
    
    @staticmethod
    def __get_cb(cb_name : str , param : Any , config : TrainConfig , model_module : Any) -> Optional[dict]:
        assert isinstance(param , dict), (cb_name , param)
        for cb_mod in _search_cb_mod:
            if hasattr(cb_mod , cb_name): 
                if cb_mod == display: param = {'verbosity': config.verbosity , **param}
                return getattr(cb_mod , cb_name)(model_module , **param)
        else: # on success
            raise KeyError(cb_name)