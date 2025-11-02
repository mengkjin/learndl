from typing import Any , Type
from importlib import import_module
from pathlib import Path

from src.proj import PATH
from src.basic import CALENDAR

class BasicUpdaterMeta(type):
    """meta class of BasicUpdater"""
    registry : dict[str , Type['BasicUpdater'] | Any] = {}
    def __new__(cls , name , bases , dct):
        new_cls = super().__new__(cls , name , bases , dct)
        if name != 'BasicUpdater':
            assert name not in cls.registry or cls.registry[name].__module__ == new_cls.__module__ , \
                f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
            assert 'update' in new_cls.__dict__  , \
                f'{name} must implement update method'
            assert 'update_rollback' in new_cls.__dict__ , \
                f'{name} must implement update_rollback method'
            assert 'recalculate_all' in new_cls.__dict__ , \
                f'{name} must implement recalculate_all method'
            cls.registry[name] = new_cls
        return new_cls

class BasicUpdater(metaclass=BasicUpdaterMeta):
    """
    base class of basic updater
    must implement update and update_rollback methods
    def update(self):
        pass
    def update_rollback(self , rollback_date : int):
        pass
    """
    _imported : bool = False
    _rollback_date : int = 99991231
    @classmethod
    def import_updaters(cls):
        if cls._imported:
            return
        paths = sorted([path for path in Path(__file__).parent.rglob('*.py') 
                        if path.is_file() and path.stem not in ['basic' , '__init__']])
        for path in paths:
            module_name = '.'.join(path.relative_to(PATH.main).with_suffix('').parts)
            import_module(module_name)
        cls._imported = True

    @classmethod
    def update(cls):
        raise NotImplementedError(f'{cls.__name__} must implement update method')

    @classmethod
    def update_rollback(cls , rollback_date : int):
        raise NotImplementedError(f'{cls.__name__} must implement update_rollback method')

    @classmethod
    def recalculate_all(cls):
        raise NotImplementedError(f'{cls.__name__} must implement recalculate_all method')

    @classmethod
    def set_rollback_date(cls , rollback_date : int):
        CALENDAR.check_rollback_date(rollback_date)
        cls._rollback_date = rollback_date