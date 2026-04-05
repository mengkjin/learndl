"""Build cross-platform commands to run Python scripts in a new terminal or subprocess."""
from __future__ import annotations
from pathlib import Path
from typing import Literal

from src.proj.util.shell_opener import Shell , ProcessDiscovery

__all__ = ['ScriptCmd']

class ScriptCmd:
    """Compose shell / OS-level invocations (macOS AppleScript, Windows/Linux shell) for a script path."""

    macos_terminal_profile_name = 'Basic'
    macos_tempfile_method = False
    def __init__(self , script : str | Path , params : dict | None = None , 
                 mode: Literal['shell', 'os'] = 'shell' , **kwargs):
        self.script = (str(script.absolute()) if isinstance(script , Path) else script)
        self.py_path = "uv run"
        # from src.proj.env import MACHINE
        # self.py_path = MACHINE.python_path
        self.params = params or {}
        assert mode in ['shell', 'os'] , f'Invalid mode: {mode}'
        self.mode = mode
        self.kwargs = kwargs

    def __repr__(self):
        return f'TerminalCmd(script={self.script}, params={self.params}, mode={self.mode})'

    def __str__(self):
        return Shell.py_cmd(self.script, py_path=self.py_path, kwargs=self.params)
    
    @property
    def real_pid(self):
        pids = ProcessDiscovery.find_running_instances(script=self.script, task_id=self.params.get('task_id',None))
        return pids[-1] if pids else None

    def run(self , as_workspace: str | None = None , from_workspace: str | None = None):
        if self.mode == 'shell':
            Shell.open_py(self.script, py_path=self.py_path, kwargs=self.params , 
                          as_workspace=as_workspace, from_workspace=from_workspace)
        else:
            Shell.run_py(self.script, py_path=self.py_path, kwargs=self.params)
        return self