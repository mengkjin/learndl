"""
Basic direct calls for the project.
"""

from __future__ import annotations
import re
from abc import ABC , abstractmethod

__all__ = ['DirectCall']

def _camel_to_snake(name : str) -> str:
    """Convert CamelCase (or mixed) identifiers to lower_snake_case."""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class DirectCall(ABC):
    """Basic direct call for the project."""
    category : str = 'Basic'
    def __init__(self , **kwargs):
        self.kwargs = kwargs
    @property
    def name(self) -> str:
        """Get the name of the direct call."""
        return self.__class__.__name__
    @property
    def snake_name(self) -> str:
        """Get the snake name of the direct call."""
        return _camel_to_snake(self.name)
    @property
    def description(self) -> str:
        """Get the description of the direct call."""
        return self.get_description(**self.kwargs)
    def __call__(self):
        return self.run()
    @abstractmethod
    def run(self) -> None:
        """Run the direct call."""
        pass
    @classmethod
    def get_description(cls , *args , **kwargs) -> str:
        """Get the description of the direct call."""
        return cls.__doc__ or ''
    @classmethod
    def go(cls , **kwargs):
        return cls(**kwargs)()
