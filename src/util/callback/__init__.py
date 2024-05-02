from typing import Any , Optional

from . import base , control , display
from .base import BasicCallBack , WithCallBack
from ..config import TrainConfig

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
        callbacks = [cls.__get_cb(k , config , model_module) for k,v in config.callbacks.items() if v > 0]
        return cls(model_module , *callbacks)
    
    @staticmethod
    def __get_cb(cb_name : str , config : TrainConfig , model_module : Any) -> Optional[dict]:
        if hasattr(control , cb_name):
            cls = getattr(control , cb_name)
            kwg = config.train_param.get('callbacks',{}).get(cb_name , {})
            return cls(model_module , **kwg)
        elif hasattr(display , cb_name):
            cls = getattr(display , cb_name)
            return cls(model_module , verbosity = config.verbosity)
        else: 
            raise KeyError(cb_name)