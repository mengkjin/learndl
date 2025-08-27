from typing import Any , Optional

from src.model.util import BaseCallBack , BaseTrainer 
from src.basic import Logger

from . import display, fit, test , nnspecific

SEARCH_MODS = [fit , display , test]
COMPULSARY_CALLBACKS = ['StatusDisplay' , 'DetailedAlphaAnalysis' , 'GroupReturnAnalysis']

class CallBackManager(BaseCallBack):
    def __init__(self , trainer , *callbacks):
        super().__init__(trainer)   
        self.callbacks : list[BaseCallBack] = [cb for cb in callbacks if isinstance(cb , BaseCallBack) and not cb.turn_off]

    def at_enter(self , hook , verbosity : int = 0):
        [cb.at_enter(hook , verbosity) for cb in self.callbacks]
    def at_exit(self, hook , verbosity : int = 0):
        [cb.at_exit(hook , verbosity) for cb in self.callbacks]

    @classmethod
    def setup(cls , trainer : BaseTrainer):
        with Logger.EnclosedMessage(' setup callbacks '):
            cb_configs = trainer.config.callbacks
            for cb in trainer.model.COMPULSARY_CALLBACKS:
                if cb not in cb_configs: cb_configs.update({cb:{}})
            
            if avail_cbs := trainer.model.AVAILABLE_CALLBACKS:
                cb_configs = {k:v for k,v in cb_configs.items() if k in avail_cbs}

            callbacks = [cls.__get_cb(cb , param , trainer) for cb , param in cb_configs.items()]
            if nn_specific_cb := nnspecific.specific_cb(trainer.config.model_module): 
                callbacks.append(nn_specific_cb(trainer))
        return cls(trainer , *callbacks)
    
    @staticmethod
    def __get_cb(cb_name : str , param : Any , trainer : BaseTrainer) -> Optional[dict]:
        assert isinstance(param , dict), (cb_name , param)
        for cb_mod in SEARCH_MODS:
            if hasattr(cb_mod , cb_name): return getattr(cb_mod , cb_name)(trainer , **param)
        else: # on success
            raise KeyError(cb_name)