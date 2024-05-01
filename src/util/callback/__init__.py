from collections import deque
from inspect import currentframe
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
    def setup(cls , model_module : Any = None , cb_names : list[str] = []):
        if model_module is None: raise Exception('model_module must be supplied')
        cbs = []
        for cb_name in cb_names:
            cb_cls = cls.__cb_class(cb_name)
            kwargs = cls.__cb_class_kwargs(cb_name , model_module.config)
            if kwargs is not None: cbs.append(cb_cls(model_module , **kwargs))
        return cls(model_module , *cbs)
    
    @classmethod
    def __cb_class(cls , cb_name : str):
        if cb_name in ['EarlyStoppage' , 'ValidationConverge' , 'TrainConverge' , 'FitConverge' , 
                       'EarlyExitRetrain' , 'NanLossRetrain' , 'ResetOptimizer' , 'DynamicDataLink']:
            return getattr(control , cb_name)
        elif cb_name in ['CallbackTimer' , 'BatchDisplay' , 'StatusDisplay']:
            return getattr(display , cb_name)
        else:
            raise KeyError(cb_name)

    @classmethod
    def __cb_class_kwargs(cls , cb_name : str , config : TrainConfig) -> Optional[dict]:
        if cb_name in ['EarlyStoppage' , 'ValidationConverge' , 'TrainConverge' , 'FitConverge' , 
                       'EarlyExitRetrain' , 'NanLossRetrain' , 'CudaEmptyCache' , 'ResetOptimizer']:
            cond = config.train_param.get('callbacks')
            kwargs = cond.get(cb_name) if isinstance(cond , dict) else None
        elif cb_name in ['CallbackTimer' , 'DynamicDataLink']:
            kwargs = {}
        elif cb_name in ['BatchDisplay' , 'StatusDisplay']:
            kwargs = {'verbosity' : config.verbosity}
        else:
            raise KeyError(cb_name)
        return {k:v for k,v in kwargs.items() if v is not None} if isinstance(kwargs , dict) else None
