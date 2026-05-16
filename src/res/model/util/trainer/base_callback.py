from __future__ import annotations

from inspect import currentframe
from typing import Any

from src.proj import Logger
from .pipeline import TrainerPipeline

class BaseCallBack(TrainerPipeline):
    CB_ORDER : int = 0
    CB_KEY_PARAMS : list[str] = []

    # Override controls, rule out others when self is used.
    OverrideCallbacks : list[str] = []

    # Conflict controls, rule out self when others are used.
    ConflictCallbacks : list[str] = []
    ConflictModuleTypes : list[str] = []
    ConflictModuleNames : list[str] = []

    def __init__(self , trainer , **kwargs) -> None:
        self.bound_with_trainer(trainer)
        self.kwargs = kwargs

    def get_info(self) -> tuple[str , str , str]:
        """return class name , class key parameters and class docstring"""
        args = {k:getattr(self , k) for k in self.CB_KEY_PARAMS}
        info = ','.join([f'{k}={v}' for k,v in args.items()])
        return self.__class__.__name__ , info , self.__class__.__doc__ or ''

    @property
    def hook_stack(self) -> list[str]:
        if not hasattr(self , '_hook_stack'):
            self._hook_stack = []
        return self._hook_stack

    def __enter__(self): 
        self.hook_stack.append(self.trace_hook_name())
        self.at_enter(self.hook_stack[-1])
    def __exit__(self , *args):
        self.at_exit(self.hook_stack.pop())
    def __bool__(self):
        return True
    def at_enter(self , hook : str , vb_level : Any = 'max'):  
        Logger.stdout(f'{hook} of callback {self.__class__.__name__} start' , vb_level = vb_level)
    def at_exit(self , hook : str , vb_level : Any = 'max'): 
        getattr(self , hook)()
        Logger.stdout(f'{hook} of callback {self.__class__.__name__} end' , vb_level = vb_level)

    def trace_hook_name(self) -> str:
        env = getattr(currentframe() , 'f_back')
        while not env.f_code.co_name.startswith('on_'): 
            env = getattr(env , 'f_back')
        return env.f_code.co_name
