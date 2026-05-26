from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from typing import Any, Callable, TypeVar, overload

from src.proj.env import Proj
from src.proj.log import Logger

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

class BaseModule:
    """
    Base module of model components, including trainer, predictor, data module, etc.
    Includes methods and properties for binder logging / vb_level and caching properties.
    """

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
    def set_vb_level(self , vb_level : Any):
        self.cached_properties.set('vb_level' , Proj.vb(vb_level))
    def set_indent(self , indent : int):
        self.cached_properties.set('indent' , indent)
    @property
    def vb_level(self) -> int:
        if not self.cached_properties.has('vb_level'):
            return getattr(self.binder, 'vb_level' , 0) + 1
        return self.cached_properties.get('vb_level')
    @property
    def indent(self) -> int:
        if not self.cached_properties.has('indent'):
            return getattr(self.binder, 'indent' , 0) + 1
        return self.cached_properties.get('indent')
    
    def stdout(self , *args , add_vb : int = 0 , add_indent : int = 0 , **kwargs):
        kwargs['vb_level'] = Proj.vb(kwargs.get('vb_level', self.vb_level) , add_vb)
        kwargs['indent'] = kwargs.get('indent', self.indent) + add_indent
        Logger.stdout(f'{self.__class__.__name__} :' , *args , **kwargs)
    def note(self , *args , add_vb : int = 0 , add_indent : int = 0 , color : str = 'lightblue' , **kwargs):
        kwargs['vb_level'] = Proj.vb(kwargs.get('vb_level', self.vb_level) , add_vb)
        kwargs['indent'] = kwargs.get('indent', self.indent) + add_indent
        Logger.stdout(f'{self.__class__.__name__} :' , *args , color = color , **kwargs)
    def alert1(self , *args , add_vb : int = 0 , add_indent : int = 0 , color : str = 'lightyellow' , to_log_file : bool = True , **kwargs):
        kwargs['vb_level'] = Proj.vb(kwargs.get('vb_level', self.vb_level) , add_vb)
        kwargs['indent'] = kwargs.get('indent', self.indent) + add_indent
        Logger.stdout(f'{self.__class__.__name__} Caution:' , *args , color = color , to_log_file = to_log_file , **kwargs)
    def alert2(self , *args , add_vb : int = 0 , add_indent : int = 0 , color : str = 'lightred' , to_log_file : bool = True , **kwargs):
        kwargs['vb_level'] = Proj.vb(kwargs.get('vb_level', 0) , add_vb)
        kwargs['indent'] = kwargs.get('indent', 0) + add_indent
        Logger.stdout(f'{self.__class__.__name__} RedAlert:' , *args , color = color , to_log_file = to_log_file , **kwargs)
    def timer(self , key : str , add_vb : int = 0 , add_indent : int = 0 , add_enter_vb : int | None = None , **kwargs):
        kwargs['vb_level'] = Proj.vb(kwargs.get('vb_level', self.vb_level) , add_vb)
        kwargs['indent'] = kwargs.get('indent', self.indent) + add_indent
        kwargs['timer_prefix'] = False
        if add_enter_vb is not None:
            kwargs['enter_vb_level'] = Proj.vb(kwargs.get('enter_vb_level', self.vb_level) , add_enter_vb)
        return Logger.Timer(f'{self.__class__.__name__} Timer({key})' , **kwargs)