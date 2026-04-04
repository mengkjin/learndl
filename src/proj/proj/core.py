"""Descriptors reading project toggles from ``configs/setting/project`` (YAML/JSON via ``MACHINE``)."""

from typing import Any
from src.proj.env import MACHINE

_project_preference = MACHINE.configs('preference' , 'project')

class ProjectPreference:
    """Descriptor: value from ``_project_settings[key]``, else ``default``."""

    def __init__(self , key : str , default : Any = None):
        self.key = key
        self.default = default

    def __get__(self , instance, owner) -> bool:
        return self.get(self.key , self.default)

    @staticmethod
    def get(key : str , default : Any = None) -> Any:
        """Read a single key from the cached project settings dict."""
        return _project_preference.get(key , default)
