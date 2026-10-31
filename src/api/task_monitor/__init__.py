"""Task monitor exports, loaded lazily so offline configuration tools stay read-only."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import TaskMonitorRepository, TaskPage, TaskSnapshot
    from .output import OutputCache

__all__ = ['OutputCache', 'TaskMonitorRepository', 'TaskPage', 'TaskSnapshot']


def __getattr__(name):
    if name == 'OutputCache':
        from .output import OutputCache
        return OutputCache
    if name in {'TaskMonitorRepository', 'TaskPage', 'TaskSnapshot'}:
        from . import core
        return getattr(core, name)
    raise AttributeError(name)
