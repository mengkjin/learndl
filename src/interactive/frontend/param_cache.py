from typing import Any, Literal

class ParamCache:
    def __init__(self):
        self.cache : dict[str, dict[str, dict[str, Any]]] = {}

    def __repr__(self):
        return f"ParamCache(cache={self.cache})"

    def get(self, script_key: str , cache_type : Literal['option', 'value', 'valid'] , name : str) -> Any:
        return self.cache.get(script_key, {}).get(cache_type, {}).get(name, None)

    def set(self, value : Any, script_key: str, cache_type : Literal['option', 'value', 'valid'] , name : str):
        if script_key not in self.cache:
            self.cache[script_key] = {}
        if cache_type not in self.cache[script_key]:
            self.cache[script_key][cache_type] = {}
        self.cache[script_key][cache_type][name] = value

    def init_script_cache(self, script_key: str):
        if script_key not in self.cache:
            self.cache[script_key] = {}
        for cache_type in ['option', 'value', 'valid']:
            if cache_type not in self.cache[script_key]:
                self.cache[script_key][cache_type] = {}

    def clear_script_cache(self, script_key: str):
        if script_key in self.cache:
            self.cache[script_key].clear()
        self.init_script_cache(script_key)

    def update_script_cache(self, script_key: str, cache_type: Literal['option', 'value', 'valid'], dict_values: dict[str, Any]):
        self.init_script_cache(script_key)
        for name, value in dict_values.items():
            self.set(value, script_key, cache_type, name)