import inspect
from typing import Type

from src.proj import Logger , Proj
from src.res.model.util import BaseCallBack , BaseTrainer 
from . import display, fit, test , nnspecific

SEARCH_MODS = [fit , display , test]

class CallBackManager(BaseCallBack):
    def __init__(self , trainer , * ,info : bool = False):
        super().__init__(trainer)   
        self.callbacks = self.get_callbacks(trainer)
        if info:
            infos = [cb.get_info() for cb in self.callbacks]
            infos_dict = {name:f'({param}); {doc}' if doc else f'({param})' for name , param , doc in infos}
            Logger.stdout_pairs(infos_dict , title = f'CallBacks Initiated:' , vb_level=2)

    def at_enter(self , hook , vb_level : int = Proj.vb.max):
        [cb.at_enter(hook , vb_level) for cb in self.callbacks]
    def at_exit(self, hook , vb_level : int = Proj.vb.max):
        [cb.at_exit(hook , vb_level) for cb in self.callbacks]

    @classmethod
    def initiate(cls , trainer : BaseTrainer , * , info = True):
        return cls(trainer , info = info)

    @classmethod
    def get_callbacks(cls , trainer : BaseTrainer) -> list[BaseCallBack]:
        available_cbs = cls.get_available_cb_names()
        if trainer.model.AVAILABLE_CALLBACKS:
            available_cbs = [cb for cb in trainer.model.AVAILABLE_CALLBACKS if cb in available_cbs]
        compulsory_cbs = trainer.model.COMPULSARY_CALLBACKS
        optional_cbs = [cb for cb in trainer.config.callbacks.keys() if cb not in compulsory_cbs and cb in available_cbs]
        use_cbs = compulsory_cbs + optional_cbs

        callback_classes = [cls.get_callback_class(cb) for cb in use_cbs]
        if specific_cb := cls.get_module_specific_callback(trainer.config.model_module):
            callback_classes.append(specific_cb)

        callback_classes = sorted(callback_classes, key=lambda x: x.CB_ORDER)
        callbacks = [cb_type(trainer , **trainer.config.callbacks.get(cb_type.__name__ , {})) for cb_type in callback_classes]
        callbacks = [cb for cb in callbacks if cb]
        return callbacks

    @classmethod
    def get_available_cb_names(cls) -> list[str]:
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