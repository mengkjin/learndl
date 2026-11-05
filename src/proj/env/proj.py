"""Non-instantiable project facade: config namespaces, verbosity, logging, and shared instances."""
from __future__ import annotations
import sys
from typing import Any , Literal , TypeAlias

from src.__version__ import __version__
from src.proj.core import Silence , NoInstanceMeta
from .machine import MACHINE
from .debug_mode import DebugMode
from .verbosity import Verbosity , VerbosityLevel
from .variable import LogWriterFile , UniqueFileList , InstanceCollection

__all__ = ['Proj']

InfoPrintType : TypeAlias = Literal['os', 'script'] | None

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
    def verbose(cls , vb_level : VerbosityLevel | None = 1):
        return cls.debug['complete_verbosity'] or not cls.vb.ignore(vb_level)

    @classmethod
    def info(cls, *, include_device: bool = True) -> dict[str, Any]:
        """Merge ``MACHINE.info()`` with verbosity and log-writer summary."""
        return {**MACHINE.info(include_device=include_device), 'Proj Verbosity' : cls.vb, 'Proj Log File' : cls.log_writer}

    @classmethod
    def print_info(cls , once_type : InfoPrintType = None , identifier = 'project_initialized'):
        """
        Print project info once per process (script: ``Proj.instances`` flag; OS: env var).

        Warns if this host is marked ``cuda_server`` but CUDA is unavailable.
        """
        from src.proj.log import Logger
        object = 'logger' if once_type == 'script' else 'os'
        # Ordinary scripts should not load Torch just to print their startup banner.
        include_device = MACHINE.cuda_server or 'torch' in sys.modules
        Logger.only_once(cls.info(include_device=include_device) , printer = 'stdout_pairs' , title = 'Project Info:' , object = object , mark = identifier)
        if MACHINE.cuda_server:
            import torch
            if not torch.cuda.is_available():
                Logger.only_once(f'[{MACHINE.name}] server should have cuda but not available, please check the cuda status' , printer = 'error' , object = object , mark = identifier)


    @classmethod
    def print_disk_info(cls):
        """Show disk space info in the best way."""
        from src.api.calls.computer import PrintDiskSpaceInfo
        PrintDiskSpaceInfo.go()
