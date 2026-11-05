"""Algorithm sub-package: NN and gradient-boost model registry.

Re-exports :class:`AlgoModule`, which provides class-method factories
``get_nn`` / ``get_boost`` and utilities for listing available models.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import AlgoModule

__all__ = ['AlgoModule']


def __getattr__(name: str):
    if name != 'AlgoModule':
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from .api import AlgoModule
    globals()[name] = AlgoModule
    return AlgoModule
