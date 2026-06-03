import subprocess
import os
import psutil

def reboot(running_pid : int | None = None):
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

    subprocess.run(["uv" , "run" , "launch.py"], cwd=os.getcwd())