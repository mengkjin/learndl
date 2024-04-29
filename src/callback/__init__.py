from inspect import currentframe
from typing import Any , Optional

from . import base , control , display , model
from .base import BasicCallBack , WithCallBack

from ..util import TrainConfig

class CallBackManager:
    def __init__(self , *callbacks):        
        self.callbacks = []
        self.with_cbs : list[WithCallBack] = []
        self.base_cbs : list[BasicCallBack] = []
        [self.add_callback(cb) for cb in callbacks]

    def add_callback(self , cb):
        self.callbacks.append(cb)
        self.with_cbs.append(cb) if self.is_withcallback(cb) else self.base_cbs.append(cb)
    
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

    @classmethod
    def setup(cls , model_module : Any = None , cb_names : list[str] = []):
        if model_module is None: raise Exception('model_module must be supplied')
        cbs = []
        for cb_name in cb_names:
            cb_cls = cls.__cb_class(cb_name)
            kwargs = cls.__cb_class_kwargs(cb_name , model_module.config)
            if kwargs is not None: cbs.append(cb_cls(model_module , **kwargs))
        return cls(*cbs)
    
    @classmethod
    def __cb_class(cls , cb_name : str):
        if cb_name in ['EarlyStoppage' , 'ValidationConverge' , 'TrainConverge' , 'FitConverge' , 
                       'EarlyExitRetrain' , 'NanLossRetrain' , 'ProcessTimer' , 'ResetOptimizer']:
            return getattr(control , cb_name)
        elif cb_name in ['DynamicDataLink']:
            return getattr(model , cb_name)
        elif cb_name in ['LoaderDisplay' , 'ProgressDisplay']:
            return getattr(display , cb_name)
        else:
            raise KeyError(cb_name)

    @classmethod
    def __cb_class_kwargs(cls , cb_name : str , config : TrainConfig) -> Optional[dict]:
        if cb_name in ['EarlyStoppage' , 'ValidationConverge' , 'TrainConverge' , 'FitConverge' , 
                       'EarlyExitRetrain' , 'NanLossRetrain' , 'CudaEmptyCache' , 'ResetOptimizer']:
            cond = config.train_param.get('callbacks')
            kwargs = cond.get(cb_name) if isinstance(cond , dict) else None
        elif cb_name in ['ProcessTimer' , 'DynamicDataLink' , 'LoaderDisplay']:
            kwargs = {}
        elif cb_name in ['ProgressDisplay']:
            kwargs = {'verbosity' : config.verbosity}
        else:
            raise KeyError(cb_name)
        return {k:v for k,v in kwargs.items() if v is not None} if isinstance(kwargs , dict) else None
