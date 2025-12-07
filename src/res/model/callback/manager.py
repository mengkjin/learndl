from typing import Type

from src.proj import Logger
from src.res.model.util import BaseCallBack , BaseTrainer 
from . import display, fit, test , nnspecific

SEARCH_MODS = [fit , display , test]

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
        with Logger.ParagraphIII('Callback Setup'):
            use_cbs = list(set(trainer.model.COMPULSARY_CALLBACKS + list(trainer.config.callbacks.keys())))
            if avail_cbs := trainer.model.AVAILABLE_CALLBACKS:
                use_cbs = [cb for cb in use_cbs if cb in avail_cbs]

            callback_types = [cls.__get_cb(cb) for cb in use_cbs]
            if specific_cb := cls.__get_specific_cb(trainer.config.model_module):
                callback_types.append(specific_cb)

            callback_types = sorted(callback_types, key=lambda x: x.CB_ORDER)
            callbacks = [cb_type(trainer , **trainer.config.callbacks.get(cb_type.__name__ , {})).print_info() for cb_type in callback_types]
        return cls(trainer , *callbacks)
    
    @staticmethod
    def __get_cb(cb_name : str) -> Type[BaseCallBack]:
        for cb_mod in SEARCH_MODS:
            if hasattr(cb_mod , cb_name): 
                cb = getattr(cb_mod , cb_name)
                assert issubclass(cb , BaseCallBack), f'{cb_name} is not a subclass of BaseCallBack'
                return cb
        else: # on success
            raise KeyError(cb_name)

    @staticmethod
    def __get_specific_cb(module_name : str) -> Type[BaseCallBack] | None:
        return nnspecific.specific_cb(module_name)