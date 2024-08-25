from typing import Any , Optional

from . import base , display, train , test , nnspecific
from ..util import BaseTrainer , TrainConfig

SEARCH_MODS = [train , display , test]
BOOSTER_AVAILABLE_CALLBACKS = ['StatusDisplay' , 'DetailedAlphaAnalysis' , 'GroupReturnAnalysis']
COMPULSARY_CALLBACKS = ['StatusDisplay' , 'DetailedAlphaAnalysis' , 'GroupReturnAnalysis']

class CallBackManager(base.CallBack):
    def __init__(self , model_module , *callbacks):
        super().__init__(model_module , with_cb=True , print_info=False)     
        self.callbacks : list[base.CallBack] = [cb for cb in callbacks if isinstance(cb , base.CallBack) and not cb.turn_off]

    def at_enter(self , hook_name):
        [cb.at_enter(hook_name) for cb in self.callbacks if cb.with_cb]
    def at_exit(self, hook_name):
        [cb.at_exit(hook_name) for cb in self.callbacks]

    @classmethod
    def setup(cls , model_module : BaseTrainer):
        config : TrainConfig = model_module.config
        cb_configs = config.callbacks
        if config.module_type in ['booster' , 'aggregator']: 
            # if the model is booster (lgbm , xgboost , catboost ...) , only use StatusDisplay and 
            cb_configs = {k:v for k,v in cb_configs.items() if k in BOOSTER_AVAILABLE_CALLBACKS}
        for cb_name in COMPULSARY_CALLBACKS:
            if cb_name not in cb_configs.keys(): cb_configs[cb_name] = {}
        callbacks = [cls.__get_cb(cb , param , model_module) for cb , param in cb_configs.items()]
        if nn_specific_cb := nnspecific.get_nn_specific_cb(config.model_module): 
            callbacks.append(nn_specific_cb(model_module))
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