import io

from src.proj.env.machine import MACHINE
from src.proj.abc import stderr

from .state import ProjStates
from . import conf as Conf

__all__ = ['Proj']

_project_settings = MACHINE.configs('proj' , 'proj_settings')

class _Verbosity:
    """
    custom verbosity
    example:
        ProjVerbosity.verbosity = 1 # for int verbosity
    """
    def __init__(self):
        self.value = _project_settings['verbosity']
    def __get__(self , instance, owner):
        return self.value
    def __set__(self, instance, value : int | None = None):
        if value is None:
            return
        assert isinstance(value , int) , f'verbosity must be an integer , got {type(value)} : {value}'
        value = max(min(value , instance.max_verbosity) , instance.min_verbosity)
        if value != self.value:
            stderr(f'Project Verbosity Changed from {self.value} to {value}' , color = 'lightred' , bold = True)
        else:
            stderr(f'Project Verbosity Unchanged at {value}' , color = 'lightred' , bold = True)
        self.value = value

class _MaxVerbosity:
    def __init__(self):
        self.value = _project_settings['max_verbosity']
    def __get__(self , instance, owner):
        return self.value

class _MinVerbosity:
    def __init__(self):
        self.value = _project_settings['min_verbosity']
    def __get__(self , instance, owner):
        return self.value

class _CallbackVbLevel:
    def __init__(self):
        self.value = _project_settings['callback_vb_level']
    def __get__(self , instance, owner):
        return self.value

class _Log_File:
    def __init__(self):
        self.value = None

    def __set__(self , instance, value):
        assert value is None or isinstance(value , io.TextIOWrapper) , f'value is not a {io.TextIOWrapper} instance: {type(value)} , cannot be set to {instance.__name__}.log_file'
        if self.value is None:
            stderr(f'Project Log File Reset to None' , color = 'lightred' , bold = True)
        else:
            stderr(f'Project Log File Set to a new file : {value.name}' , color = 'lightred' , bold = True)
        self.value = value

    def __get__(self , instance, owner):
        return self.value

class _ProjMeta(type):
    """meta class of ProjConfig"""
    verbosity = _Verbosity()
    max_verbosity = _MaxVerbosity()
    min_verbosity = _MinVerbosity()
    callback_vb_level = _CallbackVbLevel()
    log_file = _Log_File()

    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')

class Proj(metaclass=_ProjMeta):
    States = ProjStates
    Conf = Conf

    def __new__(cls , *args , **kwargs):
        raise Exception(f'{cls.__name__} cannot be instantiated')

    @classmethod
    def info(cls) -> list[str]:
        """return the machine info list"""
        return [
            *MACHINE.info(),
            f'Proj Verbosity : {cls.verbosity}', 
            f'Proj Log File  : {cls.log_file}',
            # *cls.States.info(),
        ]

    @classmethod
    def print_info(cls , script_level : bool = True , identifier = 'project_initialized'):
        """
        output project info 
        for script level or os level (only once for all scripts in one os process)
        """
        import torch , os
        from src.proj.log import Logger
        def _print_project_info():
            Logger.remark('Project Info:')
            [Logger.stdout(info , color = 'lightgreen' , bold = True) for info in cls.info()]
            if MACHINE.server and not torch.cuda.is_available():
                Logger.error(f'[{MACHINE.name}] server should have cuda but not available, please check the cuda status')

        if script_level and not getattr(cls.States , identifier , False):
            _print_project_info()
            setattr(cls.States , identifier , True)
        elif not script_level and identifier not in os.environ:
            _print_project_info()
            os.environ[identifier] = "1"
