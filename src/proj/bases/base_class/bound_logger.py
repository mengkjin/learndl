from __future__ import annotations

from contextlib import contextmanager
from functools import cached_property
from typing import Any, Callable,Literal, Sequence, TypeVar, Type

from src.proj.env import Proj
from src.proj.log import Logger , LOG_LEVEL_TYPE

T = TypeVar('T')
_MISSING = object()

__all__ = ['BoundLogger']

class ModuleLogger:
    """Module logger for the module, include most methods of Logger, and can set base indent and vb_level"""
    def __init__(self , module : type[BoundLogger] | BoundLogger):
        self.module = module
        self.idts : list[int] = []
        self.vbs : list[int] = []

    @cached_property
    def is_instance_logger(self) -> bool:
        return isinstance(self.module , BoundLogger)
    @cached_property
    def name(self) -> str:
        module = self.module.__class__ if isinstance(self.module , BoundLogger) else self.module
        return module.__name__
    
    @property
    def module_vb_level(self) -> int:
        return self.module._get_instance_vb_level() if isinstance(self.module , BoundLogger) else self.module._get_class_vb_level()
    @property
    def module_indent(self) -> int:
        return self.module._get_instance_indent() if isinstance(self.module , BoundLogger) else self.module._get_class_indent()
    @property
    def vb_level(self) -> int:
        return self.module_vb_level + sum(self.vbs)
    @property
    def indent(self) -> int:
        return self.module_indent + sum(self.idts)

    @contextmanager
    def subprocess(self , idt : int = 0  , vb = 0):
        """subprocess context manager , add idt / vb / enter_vb temporary for the subprocess"""
        self.idts.append(idt)
        self.vbs.append(vb)
        yield
        self.idts.pop()
        self.vbs.pop()

    def grep_kwargs(self , vb : int | None = None , idt : int | None = None , enter_vb : int | None = None , add_prefix : bool | None = None , **kwargs):
        if vb is not None and 'vb_level' not in kwargs:
            kwargs['vb_level'] = self.vb_level + vb
        if idt is not None and 'indent' not in kwargs:
            kwargs['indent'] = self.indent + idt
        if enter_vb is not None and 'enter_vb_level' not in kwargs:
            kwargs['enter_vb_level'] = self.vb_level + enter_vb
        if add_prefix or (add_prefix is None and (kwargs.get('indent' , 0) <= self.module_indent)):
            kwargs['prefixes'] = [f'{self.name} >>' , *kwargs.get('prefixes' , [])]
        return kwargs

    def stdout(self , *args , idt : int | None = 0 , vb : int | None = 0 , add_prefix : bool | None = None , **kwargs):
        kwargs = self.grep_kwargs(vb, idt, add_prefix = add_prefix, **kwargs)
        Logger.stdout(*args , **kwargs)

    def stdout_pairs(self , pair_list : Sequence[tuple[int , str , Any] | tuple[str , Any]] | dict[str , Any] ,
                     title : str | None = None , indent : int = 0 , vb_level : int | None = None , min_key_len : int = -1 , **kwargs):
        """
        custom stdout message of multiple pairs, each pair is a tuple of (indent , key , value) or a tuple of (key , value)
        kwargs:
            indent: add prefix '  --> ' before the message
            color , bg_color , bold: color the message
            sep , end , file , flush: same as stdout
        """
        Logger.stdout_pairs(pair_list , title = title , indent = indent , vb_level = vb_level , min_key_len = min_key_len , **kwargs)

    def caption(self , *args , idt : int | None = 0 , vb : int | None = 0 , **kwargs):
        """custom gray stdout message for caption (e.g. table / figure title)"""
        Logger.caption(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def footnote(self , *args , idt : int | None = 0 , vb : int | None = 0 , **kwargs):
        """custom gray stdout message for footnote (e.g. saved information)"""
        Logger.footnote(*args , **self.grep_kwargs(vb, idt, **kwargs))
        
    def success(self , *args , idt : int | None = 0 , vb : int | None = 0 , **kwargs):
        """custom green stdout message for success"""
        Logger.success(*args , **self.grep_kwargs(vb, idt, **kwargs))
    
    def skipping(self , *args , idt : int | None = 0 , vb : int | None = 0 , **kwargs):
        """custom skipping message"""
        Logger.skipping(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def alert1(self , *args , idt : int | None = 0 , vb : int | None = 0 , **kwargs):
        """custom stdout message with yellow for alert"""
        Logger.alert1(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def alert2(self , *args , idt : int | None = 0 , vb : int | None = 0 , **kwargs):
        """custom stdout message with lightred for alert"""
        Logger.alert2(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def alert3(self , *args , idt : int | None = 0 , vb : int | None = 0 , **kwargs):
        """custom stdout message with lightpurple for alert"""
        Logger.alert3(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def note(self , *args , idt : int | None = 0 , vb : int | None = 0 , **kwargs):
        """custom lightblue stdout message for remark"""
        Logger.note(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def remark(self , *args , vb : int | None = None , vb_level : Any = 0 , **kwargs):
        """custom lightblue stderr"""
        Logger.remark(*args , **self.grep_kwargs(vb = 0 , **kwargs))

    def debug(self , *args , vb : int | None = None , vb_level : Any = 0 , **kwargs):
        """Debug level stderr"""
        Logger.debug(*args , **self.grep_kwargs(vb = 0 , **kwargs))

    def info(self , *args , vb : int | None = None , vb_level : Any = 0 , **kwargs):
        """Info level stderr"""
        Logger.info(*args , **self.grep_kwargs(vb = 0 , **kwargs))

    def highlight(self , *args , vb : int | None = None , vb_level : Any = 0 , **kwargs):
        """custom lightcyan colored Highlight level message"""
        Logger.highlight(*args , **self.grep_kwargs(vb = 0 , **kwargs))

    def warning(self , *args , vb : int | None = None , vb_level : Any = 0 , **kwargs):
        """Warning level stderr"""
        Logger.warning(*args , **self.grep_kwargs(vb = 0 , **kwargs))

    def error(self , *args , vb : int | None = None , vb_level : Any = 0 , **kwargs):
        """Error level stderr"""
        Logger.error(*args , **self.grep_kwargs(vb = 0 , **kwargs))

    def critical(self , *args , vb : int | None = None , vb_level : Any = 0 , **kwargs):
        """Critical level stderr"""
        Logger.critical(*args , **self.grep_kwargs(vb = 0 , **kwargs))

    def only_once(self , *args , object : Any | None | Literal['os' , 'logger'] = 'logger' , mark : str = 'default' , printer : Callable | str = 'stdout' ,  **kwargs):
        """display the message only once for the same object and key"""
        Logger.only_once(*args , printer = printer , object = object , mark = mark , **kwargs)

    def log_only(self , *args , **kwargs):
        """dump to log writer with no display"""
        Logger.log_only(*args , **kwargs)

    def divider(self , *args , vb : int = 0 , **kwargs):
        """Divider mesge , use stdout"""
        Logger.divider(*args , **self.grep_kwargs(vb, **kwargs))

    def conclude(self , *args : str , **kwargs):
        """Add the message to the conclusions for later use"""
        Logger.conclude(*args , **kwargs)

    def draw_conclusions(self , simplify_errors : bool = True) -> str:
        """wrap the conclusions: printout , merge into a single string and clear them"""
        return Logger.draw_conclusions(simplify_errors = simplify_errors)

    def get_conclusions(self , type : LOG_LEVEL_TYPE) -> list[str]:
        """Get the conclusions"""
        return Logger.get_conclusions(type)

    def print_exc(self , e : Exception , color : str = 'lightred' , bold : bool = True):
        """Print the exception"""
        return Logger.print_exc(e , color = color , bold = bold)

    def print_traceback_stack(self , color : str = 'yellow' , bold : bool = True):
        """Print the exception stack"""
        return Logger.print_traceback_stack(color = color , bold = bold)

    def display(self , obj , title : str | None = None , vb : int = 0 , **kwargs):
        """
        display the object
        """
        kwargs = self.grep_kwargs(vb, add_prefix = False, **kwargs)
        Logger.display(obj , title = title , **kwargs)

    def timer(self , key : str , vb : int = 0 , idt : int = 0 , enter_vb : int | None = None , add_prefix : bool | None = None , **kwargs):
        kwargs = self.grep_kwargs(vb, idt, enter_vb, add_prefix = add_prefix, **kwargs)
        kwargs['timer_prefix'] = False
        return Logger.Timer(key , **kwargs)

    def paragraph(self , *args , vb : int = 0 , idt : int = 0 , enter_vb : int | None = None , **kwargs):
        """create a paragraph context manager"""
        kwargs = self.grep_kwargs(vb, idt, enter_vb, add_prefix = False, **kwargs)
        return Logger.Paragraph(*args , **kwargs)

class ModuleLoggerGetter:
    def __get__(self, instance : BoundLogger | None, owner : Type[BoundLogger]) -> ModuleLogger:
        if instance is None:
            if not hasattr(owner, '_cls_logger'):
                logger = ModuleLogger(owner)
                setattr(owner, '_cls_logger', logger)
            return getattr(owner, '_cls_logger')
        else:
            return instance._self_logger

class VBLevelGetter:
    def __get__(self, instance : BoundLogger | None, owner : Type[BoundLogger]) -> int:
        return instance.logger.vb_level if instance is not None else owner.logger.vb_level
class IndentGetter:
    def __get__(self, instance : BoundLogger | None, owner : Type[BoundLogger]) -> int:
        return instance.logger.indent if instance is not None else owner.logger.indent

class BoundLogger:
    """
    Bounded logger for the module, include most methods of Logger, and can set base indent and vb_level
    Both class and instance can set vb_level and indent, and the instance will inherit the class's vb_level and indent.
    For example
        class A(BoundLogger):
            ...
        a = A()

        A.SetClassVB(vb_level = 1)
        A.logger.info('hello')
        
        a.set_vb(vb_level = 2)
        a.logger.info('hello')
    """
    logger = ModuleLoggerGetter()
    vb_level = VBLevelGetter()
    indent = IndentGetter()

    def __init__(self , * , indent: int = 0 , vb_level: Any = 1 , **kwargs):
        self.set_vb(vb_level , indent)

    @property
    def binder(self) -> Any:
        if hasattr(self , '_binder'):
            return getattr(self , '_binder')
        else:
            return None

    @classmethod
    def SetClassVB(cls , vb_level : Any | None = None , indent : int | None = None):
        if vb_level is not None:
            cls._class_vb_level = Proj.vb(vb_level)
        if indent is not None:
            cls._class_indent = indent
    def set_vb(self , vb_level : Any | None = None , indent : int | None = None):
        if vb_level is not None:
            self._instance_vb_level = Proj.vb(vb_level)
        if indent is not None:
            self._instance_indent = indent
    @classmethod
    def _get_class_vb_level(cls) -> int:
        return Proj.vb(cls._class_vb_level if hasattr(cls, '_class_vb_level') else 1)
    @classmethod
    def _get_class_indent(cls) -> int:
        return cls._class_indent if hasattr(cls, '_class_indent') else 0
    def _get_instance_vb_level(self) -> int:
        if hasattr(self, '_instance_vb_level'):
            return self._instance_vb_level
        elif hasattr(self.binder, 'vb_level'):
            return getattr(self.binder, 'vb_level') + 1
        else:
            return self._get_class_vb_level() + 1
    def _get_instance_indent(self) -> int:
        if hasattr(self, '_instance_indent'):
            return self._instance_indent
        elif hasattr(self.binder, 'indent'):
            return getattr(self.binder, 'indent')
        else:
            return self._get_class_indent() + 1

    @cached_property
    def _self_logger(self) -> ModuleLogger:
        return ModuleLogger(self)