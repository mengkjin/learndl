import inspect
from typing import Type

from src.proj import Logger
from src.res.model.util import BaseCallBack , BaseTrainer 
from . import display, fit, test , nnspecific

SEARCH_MODS = [fit , display , test]

class CallBackManager(BaseCallBack):
    def __init__(self , trainer , *callbacks):
        super().__init__(trainer)   
        self.callbacks : list[BaseCallBack] = [cb for cb in callbacks if isinstance(cb , BaseCallBack) and not cb.turn_off]

    def at_enter(self , hook , vb_level : int = 10):
        [cb.at_enter(hook , vb_level) for cb in self.callbacks]
    def at_exit(self, hook , vb_level : int = 10):
        [cb.at_exit(hook , vb_level) for cb in self.callbacks]

    @classmethod
    def setup(cls , trainer : BaseTrainer):
        with Logger.ParagraphIII('Callback Setup'):
            available_cbs = cls.get_available_cbs()
            if trainer.model.AVAILABLE_CALLBACKS:
                available_cbs = [cb for cb in trainer.model.AVAILABLE_CALLBACKS if cb in available_cbs]
            compulsory_cbs = trainer.model.COMPULSARY_CALLBACKS
            optional_cbs = [cb for cb in trainer.config.callbacks.keys() if cb not in compulsory_cbs and cb in available_cbs]
            use_cbs = compulsory_cbs + optional_cbs

            callback_classes = [cls.get_callback_class(cb) for cb in use_cbs]
            if specific_cb := cls.get_module_specific_callback(trainer.config.model_module):
                callback_classes.append(specific_cb)

            callback_classes = sorted(callback_classes, key=lambda x: x.CB_ORDER)
            callbacks = [cb_type(trainer , **trainer.config.callbacks.get(cb_type.__name__ , {})).print_info() for cb_type in callback_classes]
        return cls(trainer , *callbacks)

    @classmethod
    def get_available_cbs(cls) -> list[str]:
        cbs = []
        for cb_mod in SEARCH_MODS:
            for name , obj in inspect.getmembers(cb_mod , lambda x: inspect.isclass(x) and issubclass(x , BaseCallBack)):
                cbs.append(name)
        return cbs
    
    @staticmethod
    def get_callback_class(cb_name : str) -> Type[BaseCallBack]:
        for cb_mod in SEARCH_MODS:
            if hasattr(cb_mod , cb_name): 
                cb = getattr(cb_mod , cb_name)
                assert issubclass(cb , BaseCallBack), f'{cb_name} is not a subclass of BaseCallBack'
                return cb
        else: # on success
            raise KeyError(cb_name)

    @staticmethod
    def get_module_specific_callback(module_name : str) -> Type[BaseCallBack] | None:
        return nnspecific.specific_cb(module_name)