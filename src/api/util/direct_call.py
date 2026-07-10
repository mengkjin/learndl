"""
Basic direct calls for the project.
"""

from __future__ import annotations
import os
import re
import sys
from abc import ABC , abstractmethod

from src.proj.log import Logger
from src.proj.util.shell.util import DoneActionType
from src.proj.util.cli.session import (
    ProcessQuit,
    ProcessReload,
    ProcessSpawn,
    ProcessSpawnDown,
    build_direct_call_script,
    build_exec_argv,
    can_exec_restart,
)

__all__ = ['DirectCall', 'ProcessReload', 'ProcessSpawn', 'ProcessSpawnDown', 'ProcessQuit']

def _camel_to_snake(name : str) -> str:
    """Convert CamelCase (or mixed) identifiers to lower_snake_case."""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class DirectCall(ABC):
    """Basic direct call for the project."""
    category : str = 'Basic'
    def __init__(self , **kwargs):
        self.kwargs = kwargs
    @property
    def name(self) -> str:
        """Get the name of the direct call."""
        return self.__class__.__name__
    @property
    def snake_name(self) -> str:
        """Get the snake name of the direct call."""
        return _camel_to_snake(self.name)
    @property
    def description(self) -> str:
        """Get the description of the direct call."""
        return self.get_description(**self.kwargs)
    def on_before_reload(self , reason : str) -> None:
        """Hook for cleanup before exec-restart; override in subclasses when needed."""
    def __call__(self):
        from src.proj.util.cli.magic import set_magic_spawn_handler
        set_magic_spawn_handler(lambda vertical: self._spawn_in_pane('manual /spawn', vertical=vertical))
        try:
            try:
                return self.run()
            except ProcessReload as exc:
                self._handle_reload(str(exc))
            except ProcessSpawn as exc:
                self._spawn_in_pane(str(exc))
            except ProcessSpawnDown as exc:
                self._spawn_in_pane(str(exc) , vertical=True)
            except ProcessQuit as exc:
                self._handle_quit(str(exc))
        finally:
            set_magic_spawn_handler(None)
    @abstractmethod
    def run(self) -> None:
        """Run the direct call."""
        pass
    @classmethod
    def get_description(cls , *args , **kwargs) -> str:
        """Get the description of the direct call."""
        return cls.__doc__ or ''
    @classmethod
    def exec_restart(cls , **kwargs) -> None:
        """Replace the current process with a fresh DirectCall invocation."""
        if not can_exec_restart():
            Logger.error('Cannot exec-restart inside Streamlit; use Reboot button.')
            return
        argv = build_exec_argv(cls, kwargs)
        os.execvp(argv[0], argv)

    @classmethod
    def spawn_in_pane(cls , vertical : bool = False , done_action : DoneActionType = 'pause' , **kwargs) -> None:
        """Open the same DirectCall in a new terminal pane; keep this process running."""
        if not can_exec_restart():
            Logger.error('Cannot spawn inside Streamlit; use QuickCall or Reboot button.')
            return
        from src.proj.util.shell import Shell
        script = build_direct_call_script(cls, kwargs)
        Shell.open(
            ['uv', 'run', '--frozen', 'python', '-c', script],
            done_action=done_action,
            title=cls.__name__,
            as_from_workspace='DirectCall',
            new_on='pane' if not vertical else 'pane_vertical',
        )

    @classmethod
    def spawn_restart(cls , **kwargs) -> None:
        """Alias for :meth:`spawn_in_pane` (kept for backward compatibility)."""
        cls.spawn_in_pane(**kwargs)

    def _spawn_in_pane(self , reason : str , vertical : bool = False) -> None:
        Logger.note(f'Spawning DirectCall [{self.name}] in new pane: {reason}')
        self.__class__.spawn_in_pane(**self.kwargs, vertical=vertical)

    def _handle_quit(self , reason : str) -> None:
        if not can_exec_restart():
            Logger.error('Cannot quit inside Streamlit from magic command.')
            return
        Logger.note(f'Quitting DirectCall [{self.name}]: {reason}')
        self.on_before_reload(reason)
        sys.exit(0)

    def _handle_reload(self , reason : str) -> None:
        Logger.note(f'Reloading DirectCall [{self.name}]: {reason}')
        self.on_before_reload(reason)
        self.__class__.exec_restart(**self.kwargs)

    @classmethod
    def go(cls , **kwargs):
        return cls(**kwargs)()
