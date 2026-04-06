"""Preferences for the project"""
from __future__ import annotations

from typing import Any
from src.proj.env import MACHINE

class Pref:
    """Preferences for anything in configs/preference/"""
    _preferences : dict[str , dict] = {}
    @classmethod
    def load_preference(cls, key : str , ) -> dict:
        """Load a single preference file from the cached project settings dict."""
        if key not in cls._preferences:
            cls._preferences[key] = MACHINE.configs('preference' , key)
        return cls._preferences[key]
    
    @classmethod
    def get(cls, key : str , name : str , default : Any = None) -> Any:
        """Get a single preference from the cached project settings dict."""
        pref = cls.load_preference(key)
        return pref.get(name , default)