from __future__ import annotations
import os
import psutil

from src.proj import MACHINE

def launch_app():
    from src.proj.util.shell import Shell
    cmd = 'uv run streamlit run src/interactive/main/launch.py --server.runOnSave=True'
    kwargs = {
        'done_action': 'close',
        'title': 'Streamlit Server',
        'as_from_workspace': 'Streamlit Server',
    }
    if not MACHINE.is_macos:
        kwargs['new_on'] = 'tab'
    Shell.open(cmd , cwd=os.getcwd(), **kwargs)


def kill_and_reboot_app(running_pid : int | None = None):
    if running_pid is None:
        to_kill = []
    else:
        current_process = psutil.Process(running_pid)
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

    launch_app()