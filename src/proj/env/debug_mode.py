"""Debug mode: container of debug mode contents."""
from __future__ import annotations

from collections import defaultdict

from src.proj.core import singleton
from .machine import MACHINE

__all__ = ['DebugMode']

@singleton
class DebugMode:
    """Context manager: set ``VB.vb_level`` for the block."""
    def __init__(self):
        self.debug = MACHINE.config.get('constant/project' , 'debug_mode' , default = False)
        self.contents = defaultdict(bool)

    def __bool__(self):
        return self.debug

    def __repr__(self):
        return f'DebugMode(debug={self.debug})'

    def __getitem__(self , key : str) -> bool:
        return self.debug and self.contents[key]
    
    def start(self):
        self.debug = True

    def stop(self):
        self.debug = False