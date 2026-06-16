"""
Direct calls related to application operations of this project.
"""
from __future__ import annotations
import os
import psutil

from src.proj import MACHINE
from src.api.util.direct_call import DirectCall

__all__ = ['LaunchApp' , 'KillAndRebootApp']

class LaunchApp(DirectCall):
    """Launch the streamlit app."""
    category = 'App'
    def run(self) -> None:
        from src.proj.util.shell import Shell
        cmd = 'uv run streamlit run src/api/interactive/launch.py'
        kwargs = {
            'done_action': 'pause',
            'title': 'Streamlit Server',
            'as_from_workspace': 'Streamlit Server',
        }
        if not MACHINE.is_macos:
            kwargs['new_on'] = 'tab'
        Shell.open(cmd , cwd=os.getcwd(), **kwargs)

class KillAndRebootApp(DirectCall):
    """Kill the streamlit app and reboot it."""
    category = 'App'
    def __init__(self , running_pid : int | None = None , **kwargs):
        self.kwargs = kwargs | {'running_pid': running_pid}
    @property
    def running_pid(self) -> int | None:
        return self.kwargs['running_pid']
    @classmethod
    def get_description(cls , running_pid : int | None = None , **kwargs) -> str:
        if running_pid is None:
            return f'Launch the streamlit app again. '
        return f'Kill the streamlit app (running pid: {running_pid}) and reboot it. '
    
    def run(self) -> None:
        if self.running_pid is None:
            to_kill = []
        else:
            current_process = psutil.Process(self.running_pid)
            children = current_process.children(recursive=True) 
            to_kill = [current_process] + children
        
        for proc in to_kill:
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
        gone, alive = psutil.wait_procs(to_kill, timeout=3)
        for proc in alive:
            proc.kill()
        LaunchApp.go()