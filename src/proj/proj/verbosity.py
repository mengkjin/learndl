from typing import Any , Literal
from src.proj.abc import stderr
from .abc import ProjectSetting

__all__ = ['Verbosity']

class Verbosity:
    max = ProjectSetting('vb_max' , 10)
    min = ProjectSetting('vb_min' , 0)
    never = ProjectSetting('vb_never' , 99)
    always = ProjectSetting('vb_always' , -99)
    callback = ProjectSetting('vb_level_callback' , 10)

    def __init__(self):
        assert self.never > self.max > self.min > self.always , (self.never , self.max , self.min , self.always)
        self._vb : int = ProjectSetting.get('vb' , 1)
        self._vb_level : int | None = None
        
    def __repr__(self):
        return f'{self.vb}'

    def __call__(self , value : int | None | Literal['max','min','never','always'] | Any , add_value : int = 0) -> int:
        if isinstance(value , int):
            if self.always < value < self.never:
                v = min(max(value , self.min) , self.max)
            else:
                v = value
        elif value in ['max','min','never','always']:
            v = getattr(self , value)
        elif value is None:
            v = self.vb
        else:
            raise ValueError(f'Invalid argument type: {type(value)}: {value}')
        return v + add_value

    @property
    def vb(self) -> int:
        return self._vb

    @property
    def vb_level(self) -> int | None:
        return self._vb_level

    def set_vb(self , value : int | None = None):
        if value is None:
            return
        assert isinstance(value , int) , f'verbosity must be an integer , got {type(value)} : {value}'
        value = max(min(value , self.max) , self.min)
        if value != self.vb:
            stderr(f'Project Verbosity Changed from {self.vb} to {value}' , color = 'lightred' , bold = True)
        else:
            stderr(f'Project Verbosity Unchanged at {value}' , color = 'lightred' , bold = True)
        self._vb = value

    def set_vb_level(self , value : int | None = None):
        self._vb_level = value

    class WithVbLevel:
        def __init__(self , vb_level : int | None | Literal['max','min','never','always'] | Any):
            self.vb_level = VB(vb_level)

        def __enter__(self):
            VB.set_vb_level(self.vb_level)
            return self

        def __exit__(self , exc_type , exc_value , exc_traceback):
            VB.set_vb_level(None)

    class WithVB:
        def __init__(self , vb : int | None | Literal['max','min','never','always'] | Any):
            self.vb = VB(vb)
            self.vb_prev : int | None = None

        def __enter__(self):
            self.vb_prev = VB.vb
            VB.set_vb(self.vb)
            return self

        def __exit__(self , exc_type , exc_value , exc_traceback):
            VB.set_vb(self.vb_prev)

    def ignore(self , vb_level : int | Literal['max','min','never','always'] | Any = 1):
        level = VB(vb_level)
        if level is None:
            return False
        else:
            return level >= self.never or self.vb < min(level , self.max)

    @property
    def is_max_level(self):
        return self.vb >= self.max

    def __eq__(self , other : int):
        return self.vb == other
    def __ne__(self , other : int):
        return self.vb != other
    def __lt__(self , other : int):
        return self.vb < other
    def __le__(self , other : int):
        return self.vb <= other
    def __gt__(self , other : int):
        return self.vb > other
    def __ge__(self , other : int):
        return self.vb >= other
    def __bool__(self):
        return self.vb is not None

VB = Verbosity()