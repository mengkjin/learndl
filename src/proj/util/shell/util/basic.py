"""Basic utilities for shell opener"""
from __future__ import annotations
from abc import ABC, abstractmethod

__all__ = ['BasicOpener', 'normalize_new_on']

_PANE_CAPABLE_OPENERS = frozenset({'cmux', 'wezterm'})


def normalize_new_on(new_on: str | None, opener: str) -> str | None:
    """Map ``pane`` / ``pane_vertical`` to ``tab`` when the backend cannot split panes."""
    if new_on not in ('pane', 'pane_vertical'):
        return new_on
    if opener in _PANE_CAPABLE_OPENERS:
        return new_on
    from src.proj.log import Logger
    Logger.note(f'pane spawn not supported by {opener}; falling back to tab')
    return 'tab'


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