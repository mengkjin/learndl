"""
Base metaclass and abstract base class for all custom data updaters.

``BasicCustomUpdater`` subclasses are auto-registered in ``BasicCustomUpdaterMeta.registry``
on class creation.  Concrete subclasses must implement ``update_all(update_type)``.

To register a new updater, create a file anywhere under ``update/custom/`` that
defines a subclass — ``import_updaters()`` will discover and import it automatically.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from src.proj import PATH , Base

__all__ = ['BasicCustomUpdater']

class BasicCustomUpdaterMeta(type):
    """
    Metaclass that auto-registers all ``BasicCustomUpdater`` subclasses.

    Each concrete subclass must implement ``update_all``; the metaclass enforces
    this at class creation time and raises ``AssertionError`` if it is missing.
    """
    registry : dict[str , type[BasicCustomUpdater] | Any] = {}
    def __new__(cls , name , bases , dct):
        """Create the class and register it (excluding the abstract base itself)."""
        new_cls = super().__new__(cls , name , bases , dct)
        if name != 'BasicCustomUpdater':
            assert name not in cls.registry or cls.registry[name].__module__ == new_cls.__module__ , \
                f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
            assert 'proceed_update' in new_cls.__dict__  , \
                f'{name} must implement proceed_update method'
            cls.registry[name] = new_cls
        return new_cls

class BasicCustomUpdater(Base.BasicUpdater , metaclass=BasicCustomUpdaterMeta):
    """
    base class of basic updater
    must implement proceed_update method
    """
    _imported : bool = False
    START_DATE : int = 20170101

    @classmethod
    def iter_updaters(cls) -> Iterator[type[BasicCustomUpdater]]:
        cls.import_updaters()
        for name , updater in cls.registry.items():
            yield updater

    @classmethod
    def import_updaters(cls):
        """
        Dynamically import all ``*.py`` files under the ``custom/`` directory.

        Uses ``importlib.import_module`` on each file's dotted path relative
        to ``PATH.main``.  Runs only once (guarded by ``_imported``).
        """
        if cls._imported:
            return
        paths = sorted([path for path in Path(__file__).parent.rglob('*.py')
                        if not path.name.startswith('_') and path != Path(__file__)])
        for path in paths:
            module_name = '.'.join(PATH.relative(path).with_suffix('').parts)
            import_module(module_name)
        cls._imported = True

    @classmethod
    def parse_update_input(cls , update_type : Base.UpdateType , rollback_date : int | None = None , 
        start : int | None = None , end : int | None = None , **kwargs) -> dict[str , Any]:
        if update_type == Base.UpdateType.UPDATE:
            start , end , overwrite = cls.START_DATE , None , False
        elif update_type == Base.UpdateType.ROLLBACK:
            assert rollback_date is not None , 'rollback_date is required for rollback'
            start , end , overwrite = rollback_date , None , True
        elif update_type == Base.UpdateType.RECALC:
            cls.logger.warning(f'Recalculate all {cls.__name__} is supported , but beware of the performance for {cls.__class__.__name__}!')
            assert start is not None and end is not None , 'start and end are required for recalculate'
            start , end , overwrite = max(start , cls.START_DATE) , end , True
        return {
            'start' : start , 
            'end' : end , 
            'overwrite' : overwrite , 
        }

    @classmethod
    def proceed_update(cls , start : int = START_DATE , end : int | None = None , overwrite : bool = False , **kwargs) -> Base.UpdateFlag:
        """proceed the update , input parameters are secured to be start , end , overwrite"""
        raise NotImplementedError(f'proceed_update is not implemented for {cls.__name__}')