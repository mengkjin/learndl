import sys
from datetime import datetime
from typing import Any

from src.proj.env import ProjConfig
from .silence import Silence

_ansi_color_mapping = {
    'black' : '\u001b[30m',
    'red' : '\u001b[31m',
    'green' : '\u001b[32m',
    'yellow' : '\u001b[33m',
    'blue' : '\u001b[34m',
    'purple' : '\u001b[35m',
    'cyan' : '\u001b[36m',
    'white' : '\u001b[37m',
    'gray' : '\u001b[90m',
    'lightred' : '\u001b[91m',
    'lightgreen' : '\u001b[92m',
    'lightyellow' : '\u001b[93m',
    'lightblue' : '\u001b[94m',
    'lightpurple' : '\u001b[95m',
    'lightcyan' : '\u001b[96m',
    'lightwhite' : '\u001b[97m',
}

_ansi_bg_color_mapping = {
    'black' : '\u001b[40m',
    'red' : '\u001b[41m',
    'green' : '\u001b[42m',
    'yellow' : '\u001b[43m',
    'blue' : '\u001b[44m',
    'purple' : '\u001b[45m',
    'cyan' : '\u001b[46m',
    'white' : '\u001b[47m',
    'gray' : '\u001b[100m',
    'lightred' : '\u001b[101m',
    'lightgreen' : '\u001b[102m',
    'lightyellow' : '\u001b[103m',
    'lightblue' : '\u001b[104m',
    'lightpurple' : '\u001b[105m',
    'lightcyan' : '\u001b[106m',
    'lightwhite' : '\u001b[107m',
}

_ansi_bold = '\u001b[1m'
_ansi_italic = '\u001b[3m'
_ansi_reset = '\u001b[0m'

def _ansi_styler(msg : str , color : str | None = None , bg_color : str | None = None , bold : bool = False , italic : bool = False) -> str:
    prefix = ''
    if color:
        prefix += _ansi_color_mapping.get(color , '')
    if bg_color:
        prefix += _ansi_bg_color_mapping.get(bg_color , '')
    if bold:
        prefix += _ansi_bold
    if italic:
        prefix += _ansi_italic
    suffix = _ansi_reset if prefix else ''
    return prefix + msg + suffix

class FormatStr(str):
    def __new__(cls , *args , sep = ' ' , indent : int = 0 , **kwargs):
        msg = cls.indent_str(indent) + sep.join([str(arg) for arg in args])
        self = super().__new__(cls , msg)
        return self

    def __init__(self , *args , sep = ' ' , indent : int = 0 , **kwargs):
        self.msg = self.indent_str(indent) + sep.join([str(arg) for arg in args])
        self.kwargs = kwargs
        self._level_prefix = ''

    def __str__(self):
        return self.formatted()

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.formatted())})'

    @classmethod
    def indent_str(cls , indent : int = 0) -> str:
        return '' if indent <= 0 else ('  ' * indent + '--> ')

    def formatted(self , **kwargs) -> str:
        kwargs = self.kwargs | kwargs
        color = kwargs.get('color' , None)
        bg_color = kwargs.get('bg_color' , None)
        bold = kwargs.get('bold' , False)
        italic = kwargs.get('italic' , False)
        return _ansi_styler(self.msg , color = color , bg_color = bg_color , bold = bold , italic = italic)

    def with_level_prefix(self , level: str | None = None , color : str | None = None , bg_color : str | None = None , bold : bool = True):
        if not level:
            return self
        self._level_prefix = self.level_prefix(level , color = color , bg_color = bg_color , bold = bold)
        return self

    @classmethod
    def level_prefix(cls , level: str , color : str | None = None , bg_color : str | None = None , bold : bool = True) -> str:
        msg = f'{datetime.now().strftime("%y-%m-%d %H:%M:%S")}|LEVEL:{level:9s}|'
        return _ansi_styler(msg , color = color , bg_color = bg_color , bold = bold)
        
    def write(self , stdout = False , stderr = False , file = None , end : str = '\n' , flush = False , **kwargs):
        msg = self.formatted(**kwargs) + end
        if self._level_prefix:
            msg = f'{self._level_prefix}: {msg}'
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
           sep = ' ' , end = '\n' , file = None , flush = False , vb_level : int = 1):
    """custom stdout message , vb_level can be set to control display (minimum ProjConfig.verbosity level)"""
    if Silence.silent or vb_level >= 99 or min(vb_level , ProjConfig.max_verbosity) > ProjConfig.verbosity:
        return
    fstr = FormatStr(*args , sep = sep , indent = indent , color = color , bg_color = bg_color , bold = bold , italic = italic)
    fstr.write(stdout = True , file = file , end = end , flush = flush)

def stderr(*args , indent : int = 0 , color : str | None = None , bg_color : str | None = None , bold : bool = False , italic : bool = False , 
           sep = ' ' , end = '\n' , file = None , flush = False , level_prefix : dict[str, Any] | None = None, vb_level : int = 0):
    """custom stderr message , vb_level can be set to control display (minimum ProjConfig.verbosity level)"""
    if Silence.silent or vb_level >= 99 or min(vb_level , ProjConfig.max_verbosity) > ProjConfig.verbosity:
        return
    fstr = FormatStr(*args , sep = sep , indent = indent , color = color , bg_color = bg_color , bold = bold , italic = italic)
    if level_prefix:
        fstr.with_level_prefix(**level_prefix)
    fstr.write(stderr = True , file = file , end = end , flush = flush)

    