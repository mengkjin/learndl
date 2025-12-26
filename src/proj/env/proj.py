import sys

from .state import ProjStates
from .machine import MACHINE

_default_verbosity = 2
_max_verbosity = 10
_min_verbosity = 0

class _Verbosity:
    """
    custom verbosity
    example:
        ProjVerbosity.verbosity = 1 # for int verbosity
    """
    def __init__(self):
        self._verbosity = _default_verbosity

    def __get__(self , instance, owner):
        return self._verbosity
    def __set__(self, instance, value):
        assert isinstance(value , int) , f'verbosity must be an integer , got {type(value)} : {value}'
        value = max(min(value , _max_verbosity) , _min_verbosity)
        if value != self._verbosity:
            sys.stderr.write(f'\u001b[31m\u001b[1mProject Verbosity Changed from {self._verbosity} to {value}\u001b[0m\n')
        else:
            sys.stderr.write(f'\u001b[31m\u001b[1mProject Verbosity Unchanged at {value}\u001b[0m\n')
        self._verbosity = value
        

class _ProjMeta(type):
    """meta class of ProjConfig"""
    verbosity = _Verbosity()
    max_verbosity = _max_verbosity
    min_verbosity = _min_verbosity

    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')

class Proj(metaclass=_ProjMeta):
    States = ProjStates

    def __new__(cls , *args , **kwargs):
        raise Exception(f'{cls.__name__} cannot be instantiated')

    @classmethod
    def info(cls) -> list[str]:
        """return the machine info list"""
        return [
            *MACHINE.info(),
            f'Proj Verbosity : {cls.verbosity}', 
            *cls.States.info(),
        ]

    @classmethod
    def print_info(cls , script_level : bool = True , identifier = 'project_initialized'):
        """
        print project info 
        for script level or os level (only once for all scripts in one os process)
        """
        import torch , os
        from src.proj.util import Logger
        def _print_project_info():
            [Logger.stdout(info , color = 'lightgreen') for info in cls.info()]
            if MACHINE.server and not torch.cuda.is_available():
                Logger.error(f'[{MACHINE.name}] server should have cuda but not available, please check the cuda status')

        if script_level and not getattr(cls.States , identifier , False):
            _print_project_info()
            setattr(cls.States , identifier , True)
        elif not script_level and identifier not in os.environ:
            _print_project_info()
            os.environ[identifier] = "1"
