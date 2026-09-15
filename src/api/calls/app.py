"""
Direct calls related to application operations of this project.
"""
from __future__ import annotations
import os
import psutil
import subprocess

from src.proj import MACHINE
from src.api.util.direct_call import DirectCall

__all__ = ['LaunchApp' , 'LaunchTaskMonitor' , 'ManageTaskSchedules' , 'KillAndRebootApp']

class LaunchApp(DirectCall):
    """Launch the streamlit app."""
    category = 'App'
    def run(self) -> None:
        from src.proj.util.shell import Shell
        cmd = 'uv run --frozen streamlit run src/api/interactive/launch.py'
        kwargs = {
            'done_action': 'pause',
            'title': 'Streamlit Server',
            'as_from_workspace': 'Streamlit Server',
        }
        if not MACHINE.is_macos:
            kwargs['new_on'] = 'tab'
        Shell.open(cmd , cwd=os.getcwd(), **kwargs)

class LaunchTaskMonitor(DirectCall):
    """Launch the filtered, read-only Streamlit task monitor."""
    category = 'App'
    def run(self) -> None:
        from src.proj.util.shell import Shell
        cmd = (
            'uv run --frozen streamlit run src/api/task_monitor/launch.py '
            '--server.address 127.0.0.1 --server.port 8502'
        )
        kwargs = {
            'done_action': 'pause',
            'title': 'Learndl Monitor',
            'as_from_workspace': 'Learndl Monitor',
        }
        if not MACHINE.is_macos:
            kwargs['new_on'] = 'tab'
        Shell.open(cmd , cwd=os.getcwd(), **kwargs)

class ManageTaskSchedules(DirectCall):
    """Preview, inspect, apply or roll back the unified watchdog/schedule installation."""

    category = 'App'

    def run(self) -> None:
        from src.proj import PATH, Logger

        action = self.kwargs.get('action', 'plan')
        if action not in {'plan', 'status', 'apply', 'rollback'}:
            raise ValueError(f'Unknown scheduling operation: {action}')
        command = ['bash', str(PATH.runs / 'install_schedules.sh'), action]
        if action == 'plan':
            command.append('--diff')
        # Inherit the pane's terminal so sudo can prompt and output stays visible.
        result = subprocess.run(command, cwd=PATH.main, check=False)
        if result.returncode:
            Logger.error(f'Scheduling operation [{action}] exited with code {result.returncode}; see output above.')


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
