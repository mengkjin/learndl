import inspect , itertools
from typing import Any , Type

from src.proj import Logger
from src.res.model.util import BaseCallBack , BaseTrainer 
from . import monitor, fit, test, specific

class CallBackManager(BaseCallBack):
    CallbackModules = [fit , monitor , test]

    def __init__(self , trainer , *args , **kwargs):
        super().__init__(trainer)   
        self.callbacks = self.get_callbacks(trainer)
        
    def at_enter(self , hook , vb_level : Any = 'max'):
        [cb.at_enter(hook , vb_level) for cb in self.callbacks]
    def at_exit(self, hook , vb_level : Any = 'max'):
        [cb.at_exit(hook , vb_level) for cb in self.callbacks]

    @classmethod
    def initialize(cls , trainer : BaseTrainer , * , vb_level : Any = 2 , min_key_len = -1):
        cbm = cls(trainer)
        infos = [cb.get_info() for cb in cbm.callbacks]
        infos_dict = {name:f'({param}), {doc}' if doc else f'({param})' for name , param , doc in infos}
        Logger.stdout_pairs(infos_dict , title = f'CallBacks Initiated:' , vb_level=vb_level , min_key_len = min_key_len)
        return cbm

    @classmethod
    def get_callbacks(cls , trainer : BaseTrainer) -> list[BaseCallBack]:
        callbacks = cls.get_callbacks_by_order(trainer)
        cls.check_callback_duplicates(callbacks)
        callbacks = cls.deal_callback_overrides(trainer , callbacks)
        callbacks = cls.deal_callback_conflicts(trainer , callbacks)
        return callbacks

    @classmethod
    def get_callbacks_by_order(cls , trainer : BaseTrainer) -> list[BaseCallBack]:
        available_cbs = cls.get_available_callback_names(trainer.model.AVAILABLE_CALLBACKS)
        compulsory_cbs = trainer.model.COMPULSARY_CALLBACKS + [cb for cb in trainer.config.callbackes if cb not in trainer.model.COMPULSARY_CALLBACKS]
        use_cbs = [cb for cb in compulsory_cbs if cb in available_cbs]

        callback_classes = [cls.get_callback_class(cb) for cb in use_cbs]
        if specific_cb := specific.get_specific_cb(trainer.config.model_module):
            callback_classes.append(specific_cb)

        callback_classes = sorted(callback_classes, key=lambda x: x.CB_ORDER)
        callbacks = [cb_type(trainer , **trainer.config.callback_kwargs.get(cb_type.__name__ , {})) for cb_type in callback_classes]
        callbacks = [cb for cb in callbacks if cb]
        return callbacks

    @classmethod
    def get_available_callback_names(cls , available_cbs : list[str] | None = None) -> list[str]:
        callbacks = list(cls.get_callback_classes().keys())
        return [name for name in callbacks if name in available_cbs] if available_cbs else callbacks


    @classmethod
    def check_callback_duplicates(cls , callbacks : list[BaseCallBack]):
        callback_names = [cb.__class__.__name__ for cb in callbacks]
        assert len(callback_names) == len(set(callback_names)) , f'duplicate callback names: {callback_names}'
    
    @classmethod
    def get_callback_classes(cls) -> dict[str,Type[BaseCallBack]]:
        if not hasattr(cls , '_general_callback_classes'):
            cbs = {}
            for cb_mod in cls.CallbackModules:
                for name , obj in inspect.getmembers(cb_mod , lambda x: inspect.isclass(x) and issubclass(x , BaseCallBack)):
                    if obj is BaseCallBack:
                        continue
                    assert name not in cbs , f'{name} is defined in {cb_mod} and {cbs[name].__module__}'
                    cbs[name] = obj
            cls._general_callback_classes = cbs
        return cls._general_callback_classes
    
    @classmethod
    def get_callback_class(cls , cb_name : str) -> Type[BaseCallBack]:
        return cls.get_callback_classes()[cb_name]

    @classmethod
    def deal_callback_overrides(cls , trainer : BaseTrainer , callbacks : list[BaseCallBack]) -> list[BaseCallBack]:
        callback_available = [True for _ in callbacks]
        for i , j in itertools.product(range(len(callbacks)) , range(len(callbacks))):
            if callbacks[i].__class__.__name__ in callbacks[j].OverrideCallbacks:
                Logger.alert1(f'{callbacks[j].__class__.__name__} overrides {callbacks[i].__class__.__name__}')
                callback_available[i] = False
        
        callbacks = [cb for i, cb in enumerate(callbacks) if callback_available[i]]
        return callbacks

    @classmethod
    def deal_callback_conflicts(cls , trainer : BaseTrainer , callbacks : list[BaseCallBack]) -> list[BaseCallBack]:
        module_type = trainer.config.module_type
        module_name = trainer.config.model_module
        callback_available = [True for _ in callbacks]
        for i in range(len(callbacks)):
            if module_type in callbacks[i].ConflictModuleTypes:
                Logger.alert1(f'{callbacks[i].__class__.__name__} conflicts with module type [{module_type}]')
                callback_available[i] = False
            if module_name in callbacks[i].ConflictModuleNames:
                Logger.alert1(f'{callbacks[i].__class__.__name__} conflicts with module name [{module_name}]')
                callback_available[i] = False
            for j in range(len(callbacks)):
                if callbacks[j].__class__.__name__ in callbacks[i].ConflictCallbacks:
                    Logger.alert1(f'{callbacks[i].__class__.__name__} conflicts with {callbacks[j].__class__.__name__}')
                    callback_available[i] = False
        
        callbacks = [cb for i, cb in enumerate(callbacks) if callback_available[i]]
        return callbacks