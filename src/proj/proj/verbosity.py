"""Global verbosity level: numeric ``vb``, optional per-call ``vb_level``, and context managers."""

from typing import Any , Literal
from src.proj.core import stderr
from .core import ProjectPreference

__all__ = ['Verbosity']


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
    max = ProjectPreference('vb_max' , 10)
    min = ProjectPreference('vb_min' , 0)
    never = ProjectPreference('vb_never' , 99)
    always = ProjectPreference('vb_always' , -99)
    callback = ProjectPreference('vb_level_callback' , 10)

    def __init__(self):
        """Load ``vb`` from project settings; thresholds come from the same config."""
        assert self.never > self.max > self.min > self.always , (self.never , self.max , self.min , self.always)
        self._vb : int = ProjectPreference.get('vb' , 1)
        self._vb_level : int | None = None
        
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

    @property
    def vb(self) -> int:
        """Current global verbosity (clamped to ``min``..``max`` when set)."""
        return self._vb

    @property
    def vb_level(self) -> int | None:
        """Per-context override set by ``WithVbLevel``; ``None`` means use ``vb`` only."""
        return self._vb_level

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

    def set_vb_level(self , value : int | None = None):
        """Set temporary per-thread logical level used by ``Logger`` wrappers."""
        self._vb_level = value

    class WithVbLevel:
        """Context manager: set ``VB.vb_level`` for the block."""

        def __init__(self , vb_level : int | None | Literal['max','min','never','always'] | Any):
            self.vb_level = VB(vb_level)

        def __enter__(self):
            VB.set_vb_level(self.vb_level)
            return self

        def __exit__(self , exc_type , exc_value , exc_traceback):
            VB.set_vb_level(None)

    class WithVB:
        """Context manager: temporarily replace global ``vb`` and restore on exit."""

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
        """Return True if output at ``vb_level`` should be suppressed given current ``vb``."""
        level = VB(vb_level)
        if level is None:
            return False
        else:
            return level >= self.never or self.vb < min(level , self.max)

    @property
    def is_max_level(self):
        """Whether ``vb`` is at or above ``max``."""
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