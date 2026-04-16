"""
Parameter value caching for the interactive Streamlit application.

Provides :class:`ParamCache`, a simple nested-dict store keyed by
``(script_key, cache_type, param_name)`` that persists parameter state
across Streamlit reruns within a session.
"""
from typing import Any, Literal


class NoCachedValue:
    """Sentinel type returned when a cache lookup finds no stored value."""
    ...


class ParamCache:
    """Nested dict cache for script parameter values across Streamlit reruns.

    Cache structure::

        {script_key: {cache_type: {param_name: value}}}

    where ``cache_type`` is one of ``'option'`` (widget options list),
    ``'value'`` (current widget value), or ``'valid'`` (validation result).
    """

    def __init__(self) -> None:
        """Initialise an empty cache."""
        self.cache : dict[str, dict[str, dict[str, Any]]] = {}

    def __repr__(self) -> str:
        """Return a debug string showing the full cache contents."""
        return f"ParamCache(cache={self.cache})"

    def has(self, script_key: str , cache_type : Literal['option', 'value', 'valid'] , name : str) -> bool:
        """Return True if a value is stored for the given ``(script_key, cache_type, name)``."""
        return name in self.cache.get(script_key, {}).get(cache_type, {})

    def get(self, script_key: str , cache_type : Literal['option', 'value', 'valid'] , name : str) -> Any:
        """Retrieve a cached value; raises ``KeyError`` if not present (use :meth:`has` first)."""
        return self.cache.get(script_key, {}).get(cache_type, {})[name]

    def set(self, value : Any, script_key: str, cache_type : Literal['option', 'value', 'valid'] , name : str) -> None:
        """Store *value* under the given ``(script_key, cache_type, name)`` triple."""
        if script_key not in self.cache:
            self.cache[script_key] = {}
        if cache_type not in self.cache[script_key]:
            self.cache[script_key][cache_type] = {}
        self.cache[script_key][cache_type][name] = value

    def init_script_cache(self, script_key: str) -> None:
        """Ensure all three cache-type sub-dicts exist for *script_key*."""
        if script_key not in self.cache:
            self.cache[script_key] = {}
        for cache_type in ['option', 'value', 'valid']:
            if cache_type not in self.cache[script_key]:
                self.cache[script_key][cache_type] = {}

    def clear_script_cache(self, script_key: str) -> None:
        """Clear all cached values for *script_key* and re-initialise the sub-dicts."""
        if script_key in self.cache:
            self.cache[script_key].clear()
        self.init_script_cache(script_key)

    def update_script_cache(self, script_key: str, cache_type: Literal['option', 'value', 'valid'], dict_values: dict[str, Any]) -> None:
        """Bulk-set multiple values for the given *script_key* and *cache_type*."""
        self.init_script_cache(script_key)
        for name, value in dict_values.items():
            self.set(value, script_key, cache_type, name)
