"""
Consolidate all callbacks into one
"""
from __future__ import annotations
import inspect , itertools
from collections.abc import Callable

from src.proj import Proj , Base
from src.res.model.util import BaseCallBack , BaseTrainer , ModelConfig
from . import monitor, fit, test, specific

VbLevelCallback = Proj.vb.get('callback')
CallbackModules = frozenset([fit , monitor , test])

__all__ = ['ConsolidateCallBack']

class ConsolidateCallBack(BaseCallBack):
    """consolidate all callbacks into one"""
    def __init__(self , trainer_or_config : BaseTrainer | ModelConfig , *args , **kwargs):
        super().__init__(trainer_or_config)   
        self.init_callbacks()
        
    def at_enter(self , hook , *args , **kwargs):
        self.logger.stdout(f'In Stage [{self.status.stage}], Hook {hook} start' , vb_level = VbLevelCallback)
        for cb in self.callbacks:
            if cb.is_hook_implemented(hook):
                cb.logger.stdout(f'{hook} start' , vb_level = VbLevelCallback)
                cb.at_enter(hook , *args , **kwargs)

    def at_exit(self, hook , *args , **kwargs):
        for cb in self.callbacks:
            if cb.is_hook_implemented(hook):
                cb.logger.stdout(f'{hook} end' , vb_level = VbLevelCallback)
                cb.at_exit(hook , *args , **kwargs)
        self.logger.stdout(f'In Stage [{self.status.stage}], Hook {hook} end' , vb_level = VbLevelCallback)

    def print_out(self , vb_level : Base.lit.VerbosityLevel = 2 , min_key_len = -1):
        infos = [cb.get_info() for cb in self.callbacks]
        infos_dict = {name:param for name , param , _ in infos}
        self.logger.stdout_pairs(infos_dict , title = f'CallBacks Initiated:' , vb_level = vb_level , min_key_len = min_key_len)

    def get_implemented_hook_callables(self , hook : str) -> list[Callable]:
        return [getattr(cb , hook) for cb in self.callbacks if cb.is_hook_implemented(hook)]

    @classmethod
    def initialize(cls , trainer_or_config : BaseTrainer | ModelConfig):
        return cls(trainer_or_config)

    def init_callbacks(self):
        callbacks = self.get_callbacks_by_order()
        callbacks = self.resolving_callbacks(callbacks)
        self.callbacks = callbacks

    def get_callbacks_by_order(self) -> list[BaseCallBack]:
        callback_classes = [self.get_callback_class(cb) for cb in self.config.callbackes]
        callback_classes.extend(specific.get_specific_cbs(self.config))

        callback_kwargs = self.config.callback_kwargs
        callbacks = [cls(self.binder , **callback_kwargs.get(cls.__name__ , {})) for cls in callback_classes]
        callbacks = sorted([cb for cb in callbacks if cb] , key=lambda x: x.CB_ORDER) # remove !bool(callbacks) and sort by CB_ORDER
        return callbacks

    def resolving_callbacks(self , callbacks : list[BaseCallBack]) -> list[BaseCallBack]:
        callback_available = [True for _ in callbacks]
        # resolve overrides
        for i , j in itertools.product(range(len(callbacks)) , range(len(callbacks))):
            if callbacks[i].__class__.__name__ in callbacks[j].OverrideCallbacks:
                self.logger.alert2(f'Callback [{callbacks[i].__class__.__name__}] removed, overridden by callback [{callbacks[j].__class__.__name__}]!')
                callback_available[i] = False

        # resolve conflicts
        module_type = self.config.module_type
        module_name = self.config.model_module
        for i in range(len(callbacks)):
            if module_type in callbacks[i].ConflictModuleTypes:
                self.logger.alert2(f'Callback [{callbacks[i].__class__.__name__}] removed, conflicts with module type [{module_type}]')
                callback_available[i] = False
            if module_name in callbacks[i].ConflictModuleNames:
                self.logger.alert2(f'Callback [{callbacks[i].__class__.__name__}] removed, conflicts with module name [{module_name}]')
                callback_available[i] = False
            for j in range(len(callbacks)):
                if callbacks[j].__class__.__name__ in callbacks[i].ConflictCallbacks:
                    self.logger.alert2(f'Callback [{callbacks[i].__class__.__name__}] removed, conflicts with callback [{callbacks[j].__class__.__name__}]')
                    callback_available[i] = False
        
        callbacks = [cb for i, cb in enumerate(callbacks) if callback_available[i]]
        return callbacks

    @classmethod
    def _get_callback_classes(cls) -> dict[str,type[BaseCallBack]]:
        if not hasattr(cls , '_callback_classes'):
            cbs = {}
            for cb_mod in CallbackModules:
                for name , obj in inspect.getmembers(cb_mod , lambda x: inspect.isclass(x) and issubclass(x , BaseCallBack)):
                    if obj is BaseCallBack:
                        continue
                    assert name not in cbs , f'{name} is defined in {cb_mod} and {cbs[name].__module__}'
                    cbs[name] = obj
            cls._callback_classes = cbs
        return cls._callback_classes
    
    @classmethod
    def get_callback_class(cls , cb_name : str) -> type[BaseCallBack]:
        return cls._get_callback_classes()[cb_name]