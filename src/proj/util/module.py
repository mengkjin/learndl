from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from typing import Any, Callable,Literal, Sequence, TypeVar, Type, overload

from src.proj.env import Proj
from src.proj.log import Logger , LOG_LEVEL_TYPE

T = TypeVar('T')
_MISSING = object()

__all__ = ['BaseModule']

class GroupedCachedProperties:
    """GroupedModule cached properties"""
    def __init__(self):
        self._cached_properties = defaultdict(dict)
    def __repr__(self):
        return f'{self.__class__.__name__}(cached_properties={self._cached_properties})'
    def group_keys(self , group : str) -> list[str]:
        return list(self._cached_properties[group].keys())
    def clear_all(self):
        self._cached_properties.clear()
    def clear(self , group : str):
        self._cached_properties[group].clear()
    @overload
    def set(self, key: str, value: Any, /) -> None: ...
    @overload
    def set(self, group: str, key: str, value: Any, /) -> None: ...
    def set(
        self,
        group: str,
        key: str | None = None,
        value: Any = _MISSING,
    ) -> None:
        if value is _MISSING:
            value = key  # type: ignore[assignment]
            key = group
            group = ''
        self._cached_properties[group][key] = value
    @overload
    def get(self, key: str, /) -> Any: ...
    @overload
    def get(self, group: str, key: str, /) -> Any: ...
    def get(self , group : str , key : str | None = None) -> Any:
        if key is None:
            group , key = '' , group
        try:
            return self._cached_properties[group][key]
        except KeyError:
            raise KeyError(
                f'{group} {key} not found in cached_properties, '
                f'current keys: {self.group_keys(group)}'
            )
    
    @overload
    def has(self, key: str, /) -> bool: ...
    @overload
    def has(self, group: str, key: str, /) -> bool: ...
    def has(self, group: str, key: str | None = None) -> bool:
        if key is None:
            key, group = group, ''
        return key in self._cached_properties[group]

    @overload
    def pop(self, key: str, /) -> Any: ...
    @overload
    def pop(self, group: str, key: str, /) -> Any: ...
    def pop(self, group: str, key: str | None = None) -> Any:
        if key is None:
            key, group = group, ''
        return self._cached_properties[group].pop(key)
    @overload
    def query(self, key: str, default_generator: Callable[[], T], /) -> T: ...
    @overload
    def query(self, group: str, key: str, default_generator: Callable[[], T], /) -> T: ...
    def query(
        self, group: str, key: str | Callable[[], T] , 
        default_generator: Callable[[], T] | None = None,
    ) -> T:
        if callable(key) and default_generator is None:
            default_generator = key
            group, key = '' , group
        elif callable(key):
            raise ValueError(f'Only one of key or default_generator can be callable, but got {key} and {default_generator}')

        if not self.has(group, key):
            assert default_generator is not None , f'default_generator is required when {group} {key} is not found'
            self.set(group, key, default_generator())
        return self.get(group, key)

class ModuleLogger:
    """Module logger for the module, include most methods of Logger, and can set base indent and vb_level"""
    def __init__(self , module : type[BaseModule] | BaseModule):
        self.module = module

    @property
    def name(self) -> str:
        if isinstance(self.module , BaseModule):
            return self.module.__class__.__name__
        else:
            return self.module.__name__
    @property
    def vb_level(self) -> int:
        if isinstance(self.module , BaseModule):
            return self.module.vb_level
        else:
            return BaseModule.GetClassVB()
    @property
    def indent(self) -> int:
        if isinstance(self.module , BaseModule):
            return self.module.indent
        else:
            return BaseModule.GetClassIndent()

    def grep_kwargs(self , vb : int | None = None , idt : int | None = None , enter_vb : int | None = None , no_prefix : bool = False , **kwargs):
        if vb is not None and 'vb_level' not in kwargs:
            kwargs['vb_level'] = self.vb_level + vb
        if idt is not None and 'indent' not in kwargs:
            kwargs['indent'] = self.indent + idt
        if enter_vb is not None and 'enter_vb_level' not in kwargs:
            kwargs['enter_vb_level'] = self.vb_level + enter_vb
        if not no_prefix and kwargs.get('indent' , 0) == 0:
            kwargs['prefixes'] = [f'{self.name} >>' , *kwargs.get('prefixes' , [])]
        return kwargs

    def stdout(self , *args , idt : int | None = 0 , vb : int | None = 0 , no_prefix : bool = False , **kwargs):
        kwargs = self.grep_kwargs(vb, idt, no_prefix = no_prefix, **kwargs)
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
        """custom stdout message with lightyellow for alert"""
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

    def remark(self , *args , idt : int | None = None , vb : int | None = None , **kwargs):
        """custom lightblue stderr"""
        Logger.remark(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def debug(self , *args , idt : int | None = None , vb : int | None = None , **kwargs):
        """Debug level stderr"""
        Logger.debug(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def info(self , *args , idt : int | None = None , vb : int | None = None , **kwargs):
        """Info level stderr"""
        Logger.info(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def highlight(self , *args , idt : int | None = None , vb : int | None = None , **kwargs):
        """custom lightcyan colored Highlight level message"""
        Logger.highlight(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def warning(self , *args , idt : int | None = None , vb : int | None = None , **kwargs):
        """Warning level stderr"""
        Logger.warning(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def error(self , *args , idt : int | None = None , vb : int | None = None , **kwargs):
        """Error level stderr"""
        Logger.error(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def critical(self , *args , idt : int | None = None , vb : int | None = None , **kwargs):
        """Critical level stderr"""
        Logger.critical(*args , **self.grep_kwargs(vb, idt, **kwargs))

    def only_once(self , *args , object : Any | None | Literal['os' , 'logger'] = 'logger' , mark : str = 'default' , printer : Callable | str = 'stdout' ,  **kwargs):
        """display the message only once for the same object and key"""
        Logger.only_once(*args , printer = printer , object = object , mark = mark , **kwargs)

    def log_only(self , *args , **kwargs):
        """dump to log writer with no display"""
        Logger.log_only(*args , **kwargs)

    def divider(self , *args , vb : int = 0 , **kwargs):
        """Divider mesge , use stdout"""
        kwargs = self.grep_kwargs(vb, **kwargs)
        Logger.divider(*args , **kwargs)

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

    def print_traceback_stack(self , color : str = 'lightyellow' , bold : bool = True):
        """Print the exception stack"""
        return Logger.print_traceback_stack(color = color , bold = bold)

    def display(self , obj , caption : str | None = None , vb : int = 0 , **kwargs):
        """
        display the object
        """
        kwargs = self.grep_kwargs(vb, no_prefix = True, **kwargs)
        Logger.display(obj , caption = caption , **kwargs)

    def timer(self , key : str , vb : int = 0 , idt : int = 0 , enter_vb : int | None = None , **kwargs):
        kwargs = self.grep_kwargs(vb, idt, enter_vb, no_prefix = True, **kwargs)
        kwargs['timer_prefix'] = False
        return Logger.Timer(f'{self.name}.{key} Timer' , **kwargs)

    def paragraph(self , *args , vb : int = 0 , idt : int = 0 , enter_vb : int | None = None , **kwargs):
        """create a paragraph context manager"""
        kwargs = self.grep_kwargs(vb, idt, enter_vb, no_prefix = True, **kwargs)
        return Logger.Paragraph(*args , **kwargs)

class ModuleLoggerGetter:
    def __get__(self, instance : BaseModule | None, owner : Type[BaseModule]) -> ModuleLogger:
        if instance is None:
            if not hasattr(owner, '_cls_logger'):
                logger = ModuleLogger(owner)
                setattr(owner, '_cls_logger', logger)
            return getattr(owner, '_cls_logger')
        else:
            return instance._self_logger

class BaseModule:
    """
    Base module of model components, including trainer, predictor, data module, etc.
    Includes methods and properties for binder logging / vb_level and caching properties.
    """
    logger = ModuleLoggerGetter()

    @property
    def binder(self) -> Any:
        if hasattr(self , '_binder'):
            return getattr(self , '_binder')
        else:
            return None
    @cached_property
    def cached_properties(self) -> GroupedCachedProperties:
        """
        grouped cached properties for the module
        Will store properties in miscellaneous groups. Use '' as group name for properties that only calucate once
        e.g. 'pipeline_hooks' , 'model_start' , 'data_module' , etc.
        """
        return GroupedCachedProperties()

    @classmethod
    def SetClassVB(cls , vb_level : int | None = None , indent : int | None = None):
        if vb_level is not None:
            cls._class_vb_level = Proj.vb(vb_level)
        if indent is not None:
            cls._class_indent = indent
    def set_vb(self , vb_level : int | None = None , indent : int | None = None):
        if vb_level is not None:
            self.cached_properties.set('vb_level' , Proj.vb(vb_level))
        if indent is not None:
            self.cached_properties.set('indent' , indent)
    @classmethod
    def GetClassVB(cls) -> int:
        return cls._class_vb_level if hasattr(cls, '_class_vb_level') else 1
    @classmethod
    def GetClassIndent(cls) -> int:
        return cls._class_indent if hasattr(cls, '_class_indent') else 0
    @property
    def vb_level(self) -> int:
        if self.cached_properties.has('vb_level'):
            return self.cached_properties.get('vb_level')
        elif hasattr(self.binder, 'vb_level'):
            return getattr(self.binder, 'vb_level') + 1
        else:
            return self.GetClassVB() + 1
    @property
    def indent(self) -> int:
        if self.cached_properties.has('indent'):
            return self.cached_properties.get('indent')
        elif hasattr(self.binder, 'indent'):
            return getattr(self.binder, 'indent')
        else:
            return self.GetClassIndent() + 1

    @cached_property
    def _self_logger(self) -> ModuleLogger:
        return ModuleLogger(self)