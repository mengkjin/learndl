"""Debug mode: container of debug mode contents."""
from __future__ import annotations

from src.proj.core import stderr , SingletonMeta
from .machine import MACHINE

__all__ = ['DebugMode']

class DebugMode(metaclass=SingletonMeta):
    """Context manager: set ``VB.vb_level`` for the block."""
    def __init__(self):
        self._contents = MACHINE.preference('debug')
        self._debug_mode = bool(MACHINE.preference('project' , 'debug_mode' , default = False))

    @property
    def contents(self) -> dict[str, bool]:
        return self._contents
    
    @property
    def debug_mode(self) -> bool:
        return self._debug_mode

    def __bool__(self):
        return self.debug_mode

    def __repr__(self):
        return f'DebugMode(debug={self.debug_mode})'

    def __getitem__(self , key : str) -> bool:
        return self.debug_mode and self.contents[key]
    
    def start(self):
        self._debug_mode = True
        stderr(f'Project Debug Mode Changed to True' , color = 'lightred' , bold = True)

    def stop(self):
        self._debug_mode = False
        stderr(f'Project Debug Mode Changed to False' , color = 'lightred' , bold = True)