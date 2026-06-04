"""Non-instantiable project facade: config namespaces, verbosity, logging, and shared instances."""
from __future__ import annotations
from typing import Any , Literal

from src.__version__ import __version__
from src.proj.core import Silence , NoInstanceMeta
from .machine import MACHINE
from .debug_mode import DebugMode
from .verbosity import Verbosity
from .variable import LogWriterFile , UniqueFileList , InstanceCollection

__all__ = ['Proj']

class ProjMeta(NoInstanceMeta):
    """Metaclass for ``Proj``: blocks direct instantiation and exposes module-level descriptors."""
    log_writer = LogWriterFile()
    
class Proj(metaclass=ProjMeta):
    """Static entry point: ``Conf``, ``vb``, ``instances``, paths to log writer and file lists."""
    vb = Verbosity()
    silence = Silence()
    debug = DebugMode()

    instances = InstanceCollection()
    email_attachments = UniqueFileList('email_attachments')
    exit_files = UniqueFileList('exit_files')
    version = __version__

    @classmethod
    def verbose(cls , vb_level : Literal['max','min','never','always'] | Any = 1):
        return cls.debug['complete_verbosity'] or not cls.vb.ignore(vb_level)

    @classmethod
    def info(cls) -> dict[str, Any]:
        """Merge ``MACHINE.info()`` with verbosity and log-writer summary."""
        return {**MACHINE.info(), 'Proj Verbosity' : cls.vb, 'Proj Log File' : cls.log_writer}

    @classmethod
    def print_info(cls , once_type : Literal['os' , 'script'] | None = None , identifier = 'project_initialized'):
        """
        Print project info once per process (script: ``Proj.instances`` flag; OS: env var).

        Warns if this host is marked ``cuda_server`` but CUDA is unavailable.
        """
        import torch
        from src.proj.log import Logger
        object = 'logger' if once_type == 'script' else 'os'
        Logger.only_once(cls.info() , printer = 'stdout_pairs' , title = 'Project Info:' , object = object , mark = identifier)
        if MACHINE.cuda_server and not torch.cuda.is_available():
            Logger.only_once(f'[{MACHINE.name}] server should have cuda but not available, please check the cuda status' , printer = 'error' , object = object , mark = identifier)


    @classmethod
    def print_disk_info(cls):
        """Show disk space info in the best way."""
        from src.call.computer import print_disk_space_info
        print_disk_space_info()