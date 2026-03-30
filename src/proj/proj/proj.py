"""Non-instantiable project facade: config namespaces, verbosity, logging, and shared instances."""

from typing import Any

from src.proj.env import MACHINE
from src.proj.abc import Silence

from .abc import ProjectSetting
from .verbosity import VB
from .files import LOG_WRITER , EMAIL_ATTACHMENTS , EXIT_FILES
from .ins import INSTANCES
from . import conf as Conf

__all__ = ['Proj']

class ProjMeta(type):
    """Metaclass for ``Proj``: blocks direct instantiation and exposes module-level descriptors."""

    log_writer = LOG_WRITER
    debug_mode = ProjectSetting('debug_mode')
    show_vb_level = ProjectSetting('show_vb_level')

    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')


class Proj(metaclass=ProjMeta):
    """Static entry point: ``Conf``, ``vb``, ``instances``, paths to log writer and file lists."""
    Conf = Conf
    vb = VB
    silence = Silence()
    instances = INSTANCES
    email_attachments = EMAIL_ATTACHMENTS
    exit_files = EXIT_FILES

    def __new__(cls , *args , **kwargs):
        raise Exception(f'{cls.__name__} cannot be instantiated')

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