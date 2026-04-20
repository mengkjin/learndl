"""Non-instantiable project facade: config namespaces, verbosity, logging, and shared instances."""
from __future__ import annotations
from typing import Any

from src.__version__ import __version__
from src.proj.core import Silence , NoInstanceMeta
from .machine import MACHINE
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

    debug_mode = MACHINE.config.get('constant/project' , 'debug_mode' , default = False)
    show_vb_level = MACHINE.config.get('constant/project' , 'show_vb_level' , default = False)

    instances = InstanceCollection()
    email_attachments = UniqueFileList('email_attachments')
    exit_files = UniqueFileList('exit_files')
    version = __version__

    @classmethod
    def info(cls) -> dict[str, Any]:
        """Merge ``MACHINE.info()`` with verbosity and log-writer summary."""
        return {**MACHINE.info(), 'Proj Verbosity' : cls.vb, 'Proj Log File' : cls.log_writer}

    @classmethod
    def print_info(cls , script_level : bool = True , identifier = 'project_initialized'):
        """
        Print project info once per process (script: ``Proj.instances`` flag; OS: env var).

        Warns if this host is marked ``cuda_server`` but CUDA is unavailable.
        """
        import torch , os
        from src.proj.log import Logger
        
        def _print_project_info():
            Logger.stdout_pairs(cls.info() , title = 'Project Info:')
            if MACHINE.cuda_server and not torch.cuda.is_available():
                Logger.error(f'[{MACHINE.name}] server should have cuda but not available, please check the cuda status')

        if script_level and not getattr(cls.instances , identifier , False):
            _print_project_info()
            setattr(cls.instances , identifier , True)
        elif not script_level and identifier not in os.environ:
            _print_project_info()
            os.environ[identifier] = "1"

    @classmethod
    def print_disk_info(cls):
        """Show disk space info in the best way."""
        from src.proj.util import print_disk_space_info
        print_disk_space_info()