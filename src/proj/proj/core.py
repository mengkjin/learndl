"""Descriptors reading project toggles from ``configs/setting/project`` (YAML/JSON via ``MACHINE``)."""
from __future__ import annotations
from typing import Any
from src.proj.env import MACHINE

class ProjectPreference:
    """Descriptor: value from ``_project_settings[key]``, else ``default``."""
    _preference = MACHINE.configs('preference' , 'project')

    def __init__(self , key : str , default : Any = None):
        self.key = key
        self.default = default

    def __get__(self , instance, owner) -> bool:
        return self.get(self.key , self.default)

    @classmethod
    def get(cls, key : str , default : Any = None) -> Any:
        """Read a single key from the cached project settings dict."""
        return cls._preference.get(key , default)
