from __future__ import annotations
import inspect , itertools
from typing import Any , Type , Callable

from src.proj import Logger , Proj
from src.res.model.util import BaseCallBack , BaseTrainer , ModelConfig
from . import monitor, fit, test, specific

VbLevelCallback = Proj.vb.get('callback')
CallbackModules = frozenset([fit , monitor , test])

class ConsolidateCallBack(BaseCallBack):
    '''consolidate all callbacks into one'''
    def __init__(self , trainer_or_config : BaseTrainer | ModelConfig , *args , **kwargs):
        super().__init__(trainer_or_config)   
        self.callbacks = self.get_callbacks(trainer_or_config)
        
    def at_enter(self , hook , *args , **kwargs):
        self.stdout(f'In stage [{self.status.stage}], Hook {hook} start' , vb_level = VbLevelCallback)
        for cb in self.callbacks:
            if cb.is_hook_implemented(hook):
                cb.stdout(f'{hook} start' , vb_level = VbLevelCallback)
                cb.at_enter(hook , *args , **kwargs)

    def at_exit(self, hook , *args , **kwargs):
        for cb in self.callbacks:
            if cb.is_hook_implemented(hook):
                cb.stdout(f'{hook} end' , vb_level = VbLevelCallback)
                cb.at_exit(hook , *args , **kwargs)
        self.stdout(f'In stage [{self.status.stage}], Hook {hook} end' , vb_level = VbLevelCallback)

    def print_out(self , vb_level : Any = 2 , min_key_len = -1):
        infos = [cb.get_info() for cb in self.callbacks]
        infos_dict = {name:f'({param}), {doc}' if doc else f'({param})' for name , param , doc in infos}
        Logger.stdout_pairs(infos_dict , title = f'CallBacks Initiated:' , vb_level = vb_level , min_key_len = min_key_len)

    def get_implemented_hook_callables(self , hook : str) -> list[Callable]:
        return [getattr(cb , hook) for cb in self.callbacks if cb.is_hook_implemented(hook)]

    @classmethod
    def initialize(cls , trainer_or_config : BaseTrainer | ModelConfig):
        return cls(trainer_or_config)

    @classmethod
    def get_callbacks(cls , trainer_or_config : BaseTrainer | ModelConfig) -> list[BaseCallBack]:
        callbacks = cls.get_callbacks_by_order(trainer_or_config)
        cls.check_callback_duplicates(callbacks)
        callbacks = cls.deal_callback_overrides(trainer_or_config , callbacks)
        callbacks = cls.deal_callback_conflicts(trainer_or_config , callbacks)
        return callbacks

    @classmethod
    def get_callbacks_by_order(cls , trainer_or_config : BaseTrainer | ModelConfig) -> list[BaseCallBack]:
        config = trainer_or_config.config if isinstance(trainer_or_config , BaseTrainer) else trainer_or_config
        callback_classes = [cls.get_callback_class(cb) for cb in config.callbackes]
        callback_classes.extend(specific.get_specific_cbs(config))

        callback_classes = sorted(callback_classes, key=lambda x: x.CB_ORDER)
        callbacks = [cb_class(trainer_or_config , **config.callback_kwargs.get(cb_class.__name__ , {})) for cb_class in callback_classes]
        callbacks = [cb for cb in callbacks if cb]
        return callbacks

    @classmethod
    def check_callback_duplicates(cls , callbacks : list[BaseCallBack]):
        callback_names = [cb.__class__.__name__ for cb in callbacks]
        assert len(callback_names) == len(set(callback_names)) , f'duplicate callback names: {callback_names}'
    
    @classmethod
    def get_callback_classes(cls) -> dict[str,Type[BaseCallBack]]:
        if not hasattr(cls , '_general_callback_classes'):
            cbs = {}
            for cb_mod in CallbackModules:
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
    def deal_callback_overrides(cls , trainer_or_config : BaseTrainer | ModelConfig , callbacks : list[BaseCallBack]) -> list[BaseCallBack]:
        callback_available = [True for _ in callbacks]
        for i , j in itertools.product(range(len(callbacks)) , range(len(callbacks))):
            if callbacks[i].__class__.__name__ in callbacks[j].OverrideCallbacks:
                Logger.alert2(f'Callback [{callbacks[i].__class__.__name__}] removed, overridden by callback [{callbacks[j].__class__.__name__}]!')
                callback_available[i] = False
        
        callbacks = [cb for i, cb in enumerate(callbacks) if callback_available[i]]
        return callbacks

    @classmethod
    def deal_callback_conflicts(cls , trainer_or_config : BaseTrainer | ModelConfig , callbacks : list[BaseCallBack]) -> list[BaseCallBack]:
        config = trainer_or_config.config if isinstance(trainer_or_config , BaseTrainer) else trainer_or_config
        module_type = config.module_type
        module_name = config.model_module
        callback_available = [True for _ in callbacks]
        for i in range(len(callbacks)):
            if module_type in callbacks[i].ConflictModuleTypes:
                Logger.alert2(f'Callback [{callbacks[i].__class__.__name__}] removed, conflicts with module type [{module_type}]')
                callback_available[i] = False
            if module_name in callbacks[i].ConflictModuleNames:
                Logger.alert2(f'Callback [{callbacks[i].__class__.__name__}] removed, conflicts with module name [{module_name}]')
                callback_available[i] = False
            for j in range(len(callbacks)):
                if callbacks[j].__class__.__name__ in callbacks[i].ConflictCallbacks:
                    Logger.alert2(f'Callback [{callbacks[i].__class__.__name__}] removed, conflicts with callback [{callbacks[j].__class__.__name__}]')
                    callback_available[i] = False
        
        callbacks = [cb for i, cb in enumerate(callbacks) if callback_available[i]]
        return callbacks