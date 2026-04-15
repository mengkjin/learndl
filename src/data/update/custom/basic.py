"""
Base metaclass and abstract base class for all custom data updaters.

``BasicCustomUpdater`` subclasses are auto-registered in ``BasicCustomUpdaterMeta.registry``
on class creation.  Concrete subclasses must implement ``update_all(update_type)``.

To register a new updater, create a file anywhere under ``update/custom/`` that
defines a subclass — ``import_updaters()`` will discover and import it automatically.
"""
from typing import Any , Type , Literal
from importlib import import_module
from pathlib import Path

from src.proj import PATH , Logger , CALENDAR

class BasicCustomUpdaterMeta(type):
    """
    Metaclass that auto-registers all ``BasicCustomUpdater`` subclasses.

    Each concrete subclass must implement ``update_all``; the metaclass enforces
    this at class creation time and raises ``AssertionError`` if it is missing.
    """
    registry : dict[str , Type['BasicCustomUpdater'] | Any] = {}
    def __new__(cls , name , bases , dct):
        """Create the class and register it (excluding the abstract base itself)."""
        new_cls = super().__new__(cls , name , bases , dct)
        if name != 'BasicUpdater':
            assert name not in cls.registry or cls.registry[name].__module__ == new_cls.__module__ , \
                f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
            assert 'update_all' in new_cls.__dict__  , \
                f'{name} must implement update_all method'
            cls.registry[name] = new_cls
        return new_cls

class BasicCustomUpdater(metaclass=BasicCustomUpdaterMeta):
    """
    base class of basic updater
    must implement update_all method
    def update(self):
        pass
    def rollback(self , rollback_date : int):
        pass
    """
    _imported : bool = False
    _rollback_date : int = 99991231
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
            module_name = '.'.join(path.relative_to(PATH.main).with_suffix('').parts)
            import_module(module_name)
        cls._imported = True

    @classmethod
    def update(cls):
        """Log and call ``update_all('update')``."""
        Logger.note(f'Update: {cls.__name__} since last update!')
        cls.update_all('update')

    @classmethod
    def rollback(cls , rollback_date : int):
        """Set the rollback date and call ``update_all('rollback')``."""
        Logger.note(f'Update: {cls.__name__} rollback from {rollback_date}!')
        cls.set_rollback_date(rollback_date)
        cls.update_all('rollback')

    @classmethod
    def recalculate_all(cls):
        """Log and call ``update_all('recalc')`` to force full recalculation."""
        Logger.note(f'Update: {cls.__name__} recalculate all!')
        cls.update_all('recalc')

    @classmethod
    def set_rollback_date(cls , rollback_date : int):
        """Validate and store the rollback date on the class."""
        CALENDAR.check_rollback_date(rollback_date)
        cls._rollback_date = rollback_date

    @classmethod
    def update_all(cls , update_type : Literal['recalc' , 'update' , 'rollback']):
        """
        Abstract method to be implemented by each concrete subclass.

        Called by ``update()``, ``rollback()``, and ``recalculate_all()``
        with the appropriate ``update_type`` string.
        """
        pass