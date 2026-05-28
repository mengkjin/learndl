from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from typing import Any, Callable, TypeVar, overload

T = TypeVar('T')
_MISSING = object()

__all__ = ['CachedProperties']

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

class CachedProperties:
    """
    Add a cached_properties property to the class, which is a grouped cached properties for the class.
    """
    @cached_property
    def cached_properties(self) -> GroupedCachedProperties:
        """
        grouped cached properties for the module
        Will store properties in miscellaneous groups. Use '' as group name for properties that only calucate once
        e.g. 'pipeline_hooks' , 'model_start' , 'data_module' , etc.
        """
        return GroupedCachedProperties()