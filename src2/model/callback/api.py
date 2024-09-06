from typing import Any , Optional

from . import display, train , test , nnspecific
from ..util.classes import BaseCallBack , BaseTrainer 

SEARCH_MODS = [train , display , test]
BOOSTER_AVAILABLE_CALLBACKS = ['StatusDisplay' , 'DetailedAlphaAnalysis' , 'GroupReturnAnalysis']
COMPULSARY_CALLBACKS = ['StatusDisplay' , 'DetailedAlphaAnalysis' , 'GroupReturnAnalysis']

class CallBackManager(BaseCallBack):
    def __init__(self , trainer , *callbacks):
        super().__init__(trainer)   
        self.callbacks : list[BaseCallBack] = [cb for cb in callbacks if isinstance(cb , BaseCallBack) and not cb.turn_off]

    def at_enter(self , hook_name):
        [cb.at_enter(hook_name) for cb in self.callbacks]
    def at_exit(self, hook_name):
        [cb.at_exit(hook_name) for cb in self.callbacks]

    @classmethod
    def setup(cls , trainer : BaseTrainer):
        cb_configs = trainer.config.callbacks
        if trainer.config.module_type != 'nn': 
            # if the model is booster (lgbm , xgboost , catboost ...) , only use StatusDisplay and 
            cb_configs = {k:v for k,v in cb_configs.items() if k in BOOSTER_AVAILABLE_CALLBACKS}
        for cb_name in COMPULSARY_CALLBACKS:
            if cb_name not in cb_configs.keys(): cb_configs[cb_name] = {}
        callbacks = [cls.__get_cb(cb , param , trainer) for cb , param in cb_configs.items()]
        if nn_specific_cb := nnspecific.get_nn_specific_cb(trainer.config.model_module): 
            callbacks.append(nn_specific_cb(trainer))
        return cls(trainer , *callbacks)
    
    @staticmethod
    def __get_cb(cb_name : str , param : Any , trainer : BaseTrainer) -> Optional[dict]:
        assert isinstance(param , dict), (cb_name , param)
        for cb_mod in SEARCH_MODS:
            if hasattr(cb_mod , cb_name): return getattr(cb_mod , cb_name)(trainer , **param)
        else: # on success
            raise KeyError(cb_name)