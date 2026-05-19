"""Global verbosity level: numeric ``vb``, optional per-call ``vb_level``, and context managers."""
from __future__ import annotations

from functools import cached_property
from typing import Any , Literal
from src.proj.core import stderr , singleton

from .debug_mode import DebugMode
from .machine import MACHINE

__all__ = ['Verbosity']

class WithVbLevel:
    """
    Context manager: set ``VB.vb_level`` for the block.
    Set temporary per-thread logical level used by ``Logger`` wrappers.
    """
        

    def __init__(self , vb_level : int | None | Literal['max','min','never','always'] | Any):
        self.VB = Verbosity()
        self.vb_level = self.VB(vb_level)

    def __enter__(self):
        self.VB.vb_level = self.vb_level
        return self

    def __exit__(self , exc_type , exc_value , exc_traceback):
        self.VB.vb_level = None

class WithVB:
    """Context manager: temporarily replace global ``vb`` and restore on exit."""

    def __init__(self , vb : int | None | Literal['max','min','never','always'] | Any):
        self.VB = Verbosity()
        self.vb = self.VB(vb)
        self.vb_prev : int | None = None

    def __enter__(self):
        self.vb_prev = self.VB.vb
        self.VB.set_vb(self.vb)
        return self

    def __exit__(self , exc_type , exc_value , exc_traceback):
        self.VB.set_vb(self.vb_prev)
@singleton
class Verbosity:
    """
    Verbosity level: numeric ``vb``, optional per-call ``vb_level``, and context managers.
    can be used to convert between symbolic and numeric levels.
    inputs can be int, Literal['max','min','never','always'], or None.
    
    - ``vb``: global verbosity level
    - ``vb_level``: per-call verbosity level
    - ``WithVbLevel``: context manager to temporarily set ``vb_level``
    - ``WithVB``: context manager to temporarily set ``vb``
    - ``ignore``: check if output at a given level should be suppressed
    - ``is_max_level``: check if ``vb`` is at or above ``max``
    
    """
    max = MACHINE.preference('verbosity' , 'basic/vb_max' , default = 10)
    min = MACHINE.preference('verbosity' , 'basic/vb_min' , default = 0)
    never = MACHINE.preference('verbosity' , 'basic/vb_never' , default = 99)
    always = MACHINE.preference('verbosity' , 'basic/vb_always' , default = -99)
    
    assert never > max > min > always , (never , max , min , always)

    WithVbLevel = WithVbLevel
    WithVB = WithVB
        
    def __repr__(self):
        return f'{self.vb}'

    def __call__(self , value : int | None | Literal['max','min','never','always'] | Any , add_value : int = 0) -> int:
        """Resolve a symbolic or numeric level to an int, optionally shifted by ``add_value``."""
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

    @cached_property
    def debug(self):
        """Debug mode"""
        return DebugMode()

    @property
    def vb(self) -> int:
        """Current global verbosity (clamped to ``min``..``max`` when set)."""
        if self.debug.debug_mode:
            return self.max
        if not hasattr(self , '_vb'):
            self._vb = MACHINE.preference('verbosity' , 'basic/vb' , default = 1)
        return self._vb

    @cached_property
    def vb_level(self) -> int | None:
        """Per-context override set by ``WithVbLevel``; ``None`` means use ``vb`` only."""
        return None

    def set_vb(self , value : int | None = None):
        """Persist new global ``vb``; no-op if ``value`` is ``None``."""
        if value is None:
            return
        assert isinstance(value , int) , f'verbosity must be an integer , got {type(value)} : {value}'
        value = max(min(value , self.max) , self.min)
        if value != self.vb:
            stderr(f'Project Verbosity Changed from {self.vb} to {value}' , color = 'lightred' , bold = True)
        else:
            stderr(f'Project Verbosity Unchanged at {value}' , color = 'lightred' , bold = True)
        self._vb = value

    def ignore(self , vb_level : int | Literal['max','min','never','always'] | Any = 1):
        """Return True if output at ``vb_level`` should be suppressed given current ``vb``."""
        level = self(vb_level)
        if level is None:
            return False
        else:
            return level >= self.never or self.vb < min(level , self.max)

    @property
    def is_max_level(self):
        """Whether ``vb`` is at or above ``max``."""
        return self.vb >= self.max

    def get(self , key: str) -> int:
        """Get the verbosity level for a special purpose , mostly used for differentiating debug mode and normal mode"""
        from src.proj.env.proj import Proj
        value = MACHINE.preference('verbosity' , f'special/{key}/default')
        if Proj.debug:
            value = MACHINE.preference('verbosity' , f'special/{key}/debug')
        return value

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