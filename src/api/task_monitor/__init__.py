"""Task monitor exports, loaded lazily so offline configuration tools stay read-only."""

__all__ = ['OutputCache', 'TaskMonitorRepository', 'TaskPage', 'TaskSnapshot']


def __getattr__(name):
    if name == 'OutputCache':
        from .output import OutputCache
        return OutputCache
    if name in {'TaskMonitorRepository', 'TaskPage', 'TaskSnapshot'}:
        from . import core
        return getattr(core, name)
    raise AttributeError(name)
