"""Basic utilities for shell opener"""
from __future__ import annotations

from abc import ABC , abstractmethod

__all__ = ['BasicOpener']

class BasicOpener(ABC):
    """Basic opener for shell opener"""
    def __init__(self):
        self._available = self.available()

    def __bool__(self) -> bool:
        return self._available

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(available={self._available})'

    @abstractmethod
    def available(self) -> bool:
        """Check if the opener is available"""
        pass

    @abstractmethod
    def run(self, command: str, * , cwd: str | None = None, **kwargs) -> None:
        """Run a command"""
        pass