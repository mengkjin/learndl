"""Colored terminal strings: ``FormatStr``, ``stdout``, ``stderr`` (respect ``Silence``)."""
from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

from .silence import Silence

__all__ = ['stdout' , 'stderr' , 'FormatStr']

_palette : dict[str , int] = {
    'black' : 30 ,
    'red' : 31 ,
    'green' : 32 ,
    'yellow' : 33 ,
    'blue' : 34 ,
    'purple' : 35 ,
    'cyan' : 36 ,
    'white' : 37 ,
    'gray' : 90 ,
    'lightred' : 91 ,
    'lightgreen' : 92 ,
    'lightyellow' : 93 ,
    'lightblue' : 94 ,
    'lightpurple' : 95 ,
    'lightcyan' : 96 ,
    'lightwhite' : 97 ,
}

_ansi_fg_colors : dict[str , str] = {f'{color}' : f'\u001b[{_palette[color]}m' for color in _palette}
_ansi_bg_colors : dict[str , str] = {f'{color}' : f'\u001b[{_palette[color] + 10}m' for color in _palette}
_ansi_bold = '\u001b[1m'
_ansi_italic = '\u001b[3m'
_ansi_reset = '\u001b[0m'

def _ansi_styler(msg : str , color : str | None = None , bg_color : str | None = None , bold : bool = False , italic : bool = False) -> str:
    """Wrap ``msg`` with ANSI SGR prefixes/suffix for fg, bg, bold, italic."""
    prefix = ''
    if color:
        prefix += _ansi_fg_colors.get(color , '')
    if bg_color:
        prefix += _ansi_bg_colors.get(bg_color , '')
    if bold:
        prefix += _ansi_bold
    if italic:
        prefix += _ansi_italic
    suffix = _ansi_reset if prefix else ''
    return prefix + msg + suffix

class FormatStr(str):
    """String with deferred ANSI formatting and optional timestamp/level prefix."""

    def __new__(cls , *args , sep = ' ' , indent : int = 0 , **kwargs):
        msg = cls.indent_str(indent) + sep.join([str(arg) for arg in args])
        self = super().__new__(cls , msg)
        return self

    def __init__(self , *args , sep = ' ' , indent : int = 0 , **kwargs):
        """Store raw message and style kwargs (``color``, ``bg_color``, ``bold``, ``italic``)."""
        self.msg = self.indent_str(indent) + sep.join([str(arg) for arg in args])
        self.kwargs = kwargs

    def __str__(self):
        """ANSI-formatted text (for printing)."""
        return self.formatted()

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.formatted())})'

    def __bool__(self):
        """False only when the underlying message is empty."""
        return bool(self.msg)

    @classmethod
    def empty(cls) -> FormatStr:
        if not hasattr(cls , '_empty'):
            cls._empty = cls()
        return cls._empty

    @property
    def level_prefix(self) -> FormatStr | None:
        if not hasattr(self , '_level_prefix') or self._level_prefix is None:
            return None
        return self._level_prefix

    @level_prefix.setter
    def level_prefix(self , value : FormatStr | None):
        self._level_prefix = value

    @classmethod
    def indent_str(cls , indent : int = 0) -> str:
        return '' if indent <= 0 else ('  ' * indent + '--> ')

    def formatted(self , **kwargs) -> str:
        """Apply ANSI styles; kwargs override those stored on construction."""
        kwargs = self.kwargs | kwargs
        color = kwargs.get('color' , None)
        bg_color = kwargs.get('bg_color' , None)
        bold = kwargs.get('bold' , False)
        italic = kwargs.get('italic' , False)
        new_msg = _ansi_styler(self.msg , color = color , bg_color = bg_color , bold = bold , italic = italic)
        if self.level_prefix:
            new_msg = f'{self.level_prefix.formatted()}: {new_msg}'
        return new_msg

    def unformatted(self) -> str:
        """Plain text including optional level prefix, without ANSI codes."""
        new_msg = self.msg
        if self.level_prefix:
            new_msg = f'{self.level_prefix.unformatted()}: {new_msg}'
        return new_msg

    def with_level_prefix(self , level: str | None = None , color : str | None = None , bg_color : str | None = None , bold : bool = True):
        """Prepend ``yy-mm-dd HH:MM:SS|LEVEL|`` as a nested ``FormatStr`` prefix."""
        if level:
            msg = f'{datetime.now().strftime("%y-%m-%d %H:%M:%S")}|{level:10s}|'
            self.level_prefix = FormatStr(msg , color = color , bg_color = bg_color , bold = bold)
        return self
        
    def write(self , stdout = False , stderr = False , file = None , end : str = '\n' , flush = False , **kwargs):
        """Write formatted text to stdout, stderr, and/or append to a file path."""
        msg = self.formatted(**kwargs) + end
        if file:
            with open(file , 'a') as f:
                f.write(msg)
                if flush:
                    f.flush()
        else:
            if not (stdout or stderr):
                return
            io = sys.stdout if stdout else sys.stderr
            io.write(msg)
            if flush:
                io.flush()

def stdout(*args , indent : int = 0 , color : str | None = None , bg_color : str | None = None , bold : bool = False , italic : bool = False , 
           sep = ' ' , end = '\n' , file = None , flush = False , write = True):
    """custom stdout message , vb_level can be set to control display (minimum verbosity level to show the message)"""
    if Silence.silent or not write:
        return FormatStr.empty()
    fstr = FormatStr(*args , sep = sep , indent = indent , color = color , bg_color = bg_color , bold = bold , italic = italic)
    fstr.write(stdout = True , file = file , end = end , flush = flush)
    return fstr

def stderr(*args , indent : int = 0 , color : str | None = None , bg_color : str | None = None , bold : bool = False , italic : bool = False , 
           sep = ' ' , end = '\n' , file = None , flush = False , level_prefix : dict[str, Any] | None = None, write = True):
    """Like ``stdout`` but writes to stderr; skipped when ``Silence.silent`` or ``write`` is false."""
    fstr = FormatStr(*args , sep = sep , indent = indent , color = color , bg_color = bg_color , bold = bold , italic = italic)
    if level_prefix:
        fstr.with_level_prefix(**level_prefix)
    if not Silence.silent and write:
        fstr.write(stderr = True , file = file , end = end , flush = flush)
    return fstr
    