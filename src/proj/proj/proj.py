import io
from pathlib import Path
import threading
from typing import Any

from src.proj.env.machine import MACHINE
from src.proj.abc import stderr , Silence

from .state import ProjStates
from . import conf as Conf

__all__ = ['Proj']

_project_settings = MACHINE.configs('proj' , 'proj_settings')

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

class _UniqueFileList:
    _file_lists : dict[str , list[Path]] = {}
    def __init__(self , name : str):
        self.name = name
        self.lock = threading.Lock()
        self._file_lists[self.name] = []
        self.ban_patterns = []

    def alter1(self , *args , **kwargs):
        if not hasattr(self , '_logger'):
            from src.proj.log import Logger
            self._logger = Logger
        self._logger.alert1(*args , **kwargs)

    @property
    def file_list(self):
        return self._file_lists[self.name]

    def pop_all(self):
        with self.lock:
            files = self.file_list[:]
            self.file_list.clear()
            return files

    def append(self , file : Path | str):
        with self.lock:
            file = Path(file)
            if file in self.file_list:
                return
            if any(pattern in str(file) for pattern in self.ban_patterns):
                self.alter1(f'Fail to append {file} to {self.name} due to banned patterns!' , vb_level = Proj.vb.max)
                return
            self.file_list.append(file)

    def extend(self , *files : Path | str):
        with self.lock:
            for file in files:
                file = Path(file)
                if file in self.file_list: 
                    continue
                if any(pattern in str(file) for pattern in self.ban_patterns):
                    self.alter1(f'Fail to append {file} to {self.name} due to banned patterns!' , vb_level = Proj.vb.max)
                    continue
                self.file_list.append(file)
    
    def insert(self , index : int , file : Path | str):
        with self.lock:
            file = Path(file)
            if any(pattern in str(file) for pattern in self.ban_patterns):
                self.alter1(f'Fail to insert {file} to {self.name} due to banned patterns!' , vb_level = Proj.vb.max)
                return
            if file in self.file_list:
                self.file_list.remove(file)
            self.file_list.insert(index , file)

    def remove(self , file : Path | str):
        with self.lock:
            self.file_list.remove(Path(file))

    def ban(self , *patterns : str):
        with self.lock:
            self.ban_patterns.extend(patterns)

    def unban(self , *patterns : str):
        with self.lock:
            self.ban_patterns = [pattern for pattern in self.ban_patterns if pattern not in patterns]

    def exclude(self , *patterns : str):
        with self.lock:
            for file in self.file_list[:]:
                if any(pattern in str(file) for pattern in patterns):
                    self.file_list.remove(file)
                    self.alter1(f'Removed file {file} from {self.name} due to banned patterns!' , vb_level = Proj.vb.max)

class _Verbosity:
    max : int = _project_settings.get('vb_max' , 10)
    min : int = _project_settings.get('vb_min' , 0)
    inf : int = _project_settings.get('vb_inf' , 99)
    level_callback : int = _project_settings.get('vb_level_callback' , 10)
        
    def __init__(self):
        self.vb : int = _project_settings.get('vb' , 1)
        self.vb_level : int | None = None
        
    def __repr__(self):
        return f'{self.vb}'

    class WithVbLevel:
        def __init__(self , vb_level : int | None):
            self.vb_level = vb_level

        def __enter__(self):
            Proj.vb.vb_level = self.vb_level
            return self

        def __exit__(self , exc_type , exc_value , exc_traceback):
            Proj.vb.vb_level = None

    class WithVB:
        def __init__(self , vb : int | None = None):
            self.vb = vb
            self.vb_prev : int | None = None

        def __enter__(self):
            self.vb_prev = Proj.vb.vb
            Proj.vb.set_vb(self.vb)
            return self

        def __exit__(self , exc_type , exc_value , exc_traceback):
            Proj.vb.set_vb(self.vb_prev)

    Silence = Silence

    def set_vb(self , value : int | None = None):
        if value is None:
            return
        assert isinstance(value , int) , f'verbosity must be an integer , got {type(value)} : {value}'
        value = max(min(value , self.max) , self.min)
        if value != self.vb:
            stderr(f'Project Verbosity Changed from {self.vb} to {value}' , color = 'lightred' , bold = True)
        else:
            stderr(f'Project Verbosity Unchanged at {value}' , color = 'lightred' , bold = True)
        self.vb = value

    def ignore(self , vb_level : int = 1):
        return vb_level >= self.inf or self.vb < min(vb_level , self.max)

    @property
    def is_max_level(self):
        return self.vb >= self.max

    def __eq__(self , other : int):
        return self.vb == other
    def __ne__(self , other : int):
        return self.vb != other
    def __lt__(self , other : int):
        return self.vb < other
    def __le__(self , other : int):
        return self.vb <= other
    def __gt__(self , other : int):
        return self.vb > other
    def __ge__(self , other : int):
        return self.vb >= other
    def __bool__(self):
        return self.vb is not None

class _ProjMeta(type):
    """meta class of ProjConfig"""
    log_file = _Log_File()

    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')

class Proj(metaclass=_ProjMeta):
    States = ProjStates
    Conf = Conf
    vb = _Verbosity()
    email_attachments = _UniqueFileList('email_attachments')
    exit_files = _UniqueFileList('exit_files')

    def __new__(cls , *args , **kwargs):
        raise Exception(f'{cls.__name__} cannot be instantiated')

    @classmethod
    def info(cls) -> dict[str, Any]:
        """return the machine info list"""
        return {**MACHINE.info(), 'Proj Verbosity' : cls.vb, 'Proj Log File' : cls.log_file}

    @classmethod
    def print_info(cls , script_level : bool = True , identifier = 'project_initialized'):
        """
        output project info 
        for script level or os level (only once for all scripts in one os process)
        """
        import torch , os
        from src.proj.log import Logger
        def _print_project_info():
            Logger.stdout_pairs(cls.info() , title = 'Project Info:')
            if MACHINE.cuda_server and not torch.cuda.is_available():
                Logger.error(f'[{MACHINE.name}] server should have cuda but not available, please check the cuda status')

        if script_level and not getattr(cls.States , identifier , False):
            _print_project_info()
            setattr(cls.States , identifier , True)
        elif not script_level and identifier not in os.environ:
            _print_project_info()
            os.environ[identifier] = "1"

    @classmethod
    def print_disk_info(cls):
        from src.proj.func.disk_info import print_disk_space_info
        print_disk_space_info()