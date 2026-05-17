from __future__ import annotations

from inspect import currentframe

from src.proj import Proj
from .pipeline import TrainerPipeline

vb_level_callback = Proj.vb.get('callback')

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
    def at_enter(self , hook : str , *args , **kwargs):  
        ...
    def at_exit(self , hook : str , *args , **kwargs): 
        self.execute_hook(hook , *args , **kwargs)

    def trace_hook_name(self) -> str:
        env = getattr(currentframe() , 'f_back')
        while not env.f_code.co_name.startswith('on_'): 
            env = getattr(env , 'f_back')
        return env.f_code.co_name
