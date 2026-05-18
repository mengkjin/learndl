from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, TypeVar
from src.proj import Proj , Logger

T = TypeVar('T')
class ModuleCachedProperties:
    """Module cached properties"""
    def __init__(self):
        self._cached_properties = defaultdict(dict)
    def __repr__(self):
        return f'ModuleCachedProperties(cached_properties={self._cached_properties})'
    def get_group(self , group : str) -> dict[str,Any]:
        return self._cached_properties[group]
    def group_keys(self , group : str) -> list[str]:
        return list(self._cached_properties[group].keys())
    def set_value(self , group : str , key : str , value : Any):
        self._cached_properties[group][key] = value
    def get_value(self , group : str , key : str) -> Any:
        try:
            return self._cached_properties[group][key]
        except KeyError:
            raise KeyError(f'{group} {key} not found in cached_properties, current keys: {self.group_keys(group)}')
    def clear_all(self):
        self._cached_properties.clear()
    def clear(self , group : str):
        self._cached_properties[group].clear()
    def pop(self , group : str , key : str):
        self._cached_properties[group].pop(key)
    def has(self , group : str , key : str) -> bool:
        return key in self._cached_properties[group]
    def get(self , group : str , key : str , default_generator : Callable[[],T] | None = None , *args , **kwargs) -> T:
        if not self.has(group , key) and default_generator is not None:
            self.set_value(group , key , default_generator(*args , **kwargs))
        return self.get_value(group , key)

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
    def set_vb_level(self , vb_level : Any):
        self._vb_level = Proj.vb(vb_level)
    @property
    def vb_level(self) -> int:
        if not hasattr(self , '_vb_level'):
            return getattr(self.binder, 'vb_level' , 0) + 1
        return self._vb_level
    
    def stdout(self , *args , add_vb : int = 0 , **kwargs):
        kwargs['vb_level'] = Proj.vb(kwargs.get('vb_level', self.vb_level) , add_vb)
        Logger.stdout(f'{self.__class__.__name__} :' , *args , **kwargs)
    def note(self , *args , add_vb : int = 0 , color : str = 'lightblue' , **kwargs):
        kwargs['vb_level'] = Proj.vb(kwargs.get('vb_level', self.vb_level) , add_vb)
        Logger.stdout(f'{self.__class__.__name__} :' , *args , color = color , **kwargs)
    def alert1(self , *args , add_vb : int = 0 , color : str = 'lightyellow' , to_log_file : bool = True , **kwargs):
        kwargs['vb_level'] = Proj.vb(kwargs.get('vb_level', self.vb_level) , add_vb)
        Logger.stdout(f'{self.__class__.__name__} Caution:' , *args , color = color , to_log_file = to_log_file , **kwargs)
    def alert2(self , *args , add_vb : int = 0 , color : str = 'lightred' , to_log_file : bool = True , **kwargs):
        kwargs['vb_level'] = Proj.vb(kwargs.get('vb_level', 0) , add_vb)
        Logger.stdout(f'{self.__class__.__name__} RedAlert:' , *args , color = color , to_log_file = to_log_file , **kwargs)
    
    # @property
    # def cached_properties(self) -> dict[str,dict[str,Any]]:
    #     if not hasattr(self , '_cached_properties'):
    #         self._cached_properties = defaultdict(dict[str,Any])
    #     return self._cached_properties
    @property
    def cached_properties(self) -> ModuleCachedProperties:
        if not hasattr(self , '_cached_properties'):
            self._cached_properties = ModuleCachedProperties()
        return self._cached_properties