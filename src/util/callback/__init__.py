from typing import Any , Optional

from . import base , control , display
from .nnspecific import get_nn_specific_cb
from .base import CallBack
from ..config import TrainConfig
from ...classes import BaseTrainer
from ...environ import BOOSTER_MODULE

SEARCH_MODS = [control , display]

class CallBackManager(CallBack):
    def __init__(self , model_module , *callbacks):
        super().__init__(model_module , with_cb=True , print_info=False)     
        self.callbacks : list[CallBack] = [cb for cb in callbacks if isinstance(cb , CallBack)]

    def at_enter(self , hook_name):
        [cb.at_enter(hook_name) for cb in self.callbacks if cb.with_cb]
    def at_exit(self, hook_name):
        [cb.at_exit(hook_name) for cb in self.callbacks]

    @classmethod
    def setup(cls , model_module : BaseTrainer):
        config : TrainConfig = model_module.config
        cb_configs = config.callbacks
        if config.model_module in BOOSTER_MODULE: cb_configs = {k:v for k,v in cb_configs.items() if k in ['StatusDisplay']}
        callbacks = [cls.__get_cb(cb , param , model_module) for cb , param in cb_configs.items()]
        callbacks.append(get_nn_specific_cb(config.model_module)(model_module))
        return cls(model_module , *callbacks)
    
    @staticmethod
    def __get_cb(cb_name : str , param : Any , model_module : BaseTrainer) -> Optional[dict]:
        assert isinstance(param , dict), (cb_name , param)
        config : TrainConfig = model_module.config
        for cb_mod in SEARCH_MODS:
            if hasattr(cb_mod , cb_name): 
                if cb_mod == display: param = {'verbosity': config.verbosity , **param}
                return getattr(cb_mod , cb_name)(model_module , **param)
        else: # on success
            raise KeyError(cb_name)