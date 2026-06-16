"""Global verbosity level: numeric ``vb``, optional per-call ``vb_level``, and context managers."""
from __future__ import annotations

import numpy as np
from contextlib import contextmanager
from functools import cached_property
from typing import Any

from src.proj.core import stderr , SingletonMeta
from src.proj.core.literals import VerbosityLevel
from .debug_mode import DebugMode
from .machine import MACHINE

__all__ = ['Verbosity']

class Verbosity(metaclass=SingletonMeta):
    """
    Verbosity level: numeric ``vb``, optional per-call ``vb_level``, and context managers.
    can be used to convert between symbolic and numeric levels.
    inputs can be int or 'max' or 'min' or 'never' or 'always' or None.
    
    - ``vb``: global verbosity level
    - ``vb_level``: per-call verbosity level
    - ``ignore``: check if output at a given level should be suppressed
    - ``is_max_level``: check if ``vb`` is at or above ``max``

    context managers:
        - ``temporary_vb``: temporarily set the vb
        - ``record_vb_level``: record the vb_level
        - ``subprocess_vb``: temporarily set the additive vb_level and indent for subprocess
    
    """
    max = MACHINE.preference('verbosity' , 'basic/vb_max' , default = 10)
    min = MACHINE.preference('verbosity' , 'basic/vb_min' , default = 0)
    never = MACHINE.preference('verbosity' , 'basic/vb_never' , default = 99)
    always = MACHINE.preference('verbosity' , 'basic/vb_always' , default = -99)
    
    assert never > max > min > always , (never , max , min , always)
        
    def __repr__(self):
        return f'{self.vb}'

    def __call__(self , value : VerbosityLevel | None = None , add_value : int = 0) -> int:
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

    @property
    def vb_level(self) -> int | None:
        """Per-context override set by ``RecordVbLevel``; ``None`` means use ``vb`` only."""
        if not hasattr(self , '_vb_level'):
            self._vb_level = None
        return self._vb_level

    @property
    def additive_vbs(self) -> np.ndarray:
        """Additive vbs for subprocess"""
        if not hasattr(self , '_additive_vbs'):
            self._additive_vbs : list[np.ndarray] = []
        value : np.ndarray | Any = sum(self._additive_vbs) if self._additive_vbs else np.array([0 , 0])
        return value

    @property
    def add_vb_level(self) -> int:
        """Add vb_level for subprocess"""
        return self.additive_vbs[0]

    @property
    def add_indent(self) -> int:
        """Add indent for subprocess"""
        return self.additive_vbs[1]

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

    def ignore(self , vb_level : VerbosityLevel | None = 1):
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

    @contextmanager
    def record_vb_level(self , vb_level : VerbosityLevel | None = 1):
        """Context manager: Record the vb_level"""
        _ = self.vb_level
        old_vb_level = self._vb_level
        try:
            self._vb_level = self(vb_level)
            yield
        finally:
            self._vb_level = old_vb_level

    @contextmanager
    def temporary_vb(self, vb: VerbosityLevel | None = None):
        """Context manager: Temporarily set the vb"""
        _ = self.vb
        old_vb = self._vb
        try:
            self.set_vb(self(vb))
            yield
        finally:
            self.set_vb(old_vb)

    @contextmanager
    def subprocess_vb(self , vb : int = 0 , idt : int = 0):
        """Context manager: Temporarily set the add_vb_level (vb) and add_indent (idt) for subprocess"""
        _ = self.additive_vbs
        add_value = np.array([vb , idt])
        try:
            self._additive_vbs.append(add_value)
            yield
        finally:
            self._additive_vbs.remove(add_value)

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