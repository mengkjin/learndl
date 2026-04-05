"""Process discovery, power profile (non-Windows), and CLI arg parsing helpers."""

import fnmatch , psutil , subprocess , time , argparse
from datetime import datetime
from pathlib import Path
from typing import Any , Literal

from src.proj.env import MACHINE
from src.proj.log import Logger

__all__ = ['get_running_scripts' , 'change_power_mode' , 'check_process_status' , 'kill_process' , 'argparse_dict' , 'unknown_args']

def get_running_scripts(exclude_scripts : list[str] | str | None = None , script_type = ['*.py'] , default_excludes = ['kernel_interrupt_daemon.py']):
    """List script paths from running processes whose cmdline matches ``script_type``."""
    running_scripts : list[Path] = []
    if isinstance(exclude_scripts , str): 
        exclude_scripts = [exclude_scripts]
    excludes = [Path(scp).name for scp in (exclude_scripts or []) + default_excludes]
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline: 
                continue
            for line in cmdline:
                if any(fnmatch.fnmatch(line, pattern) for pattern in script_type):
                    if any(scp in line for scp in excludes): 
                        pass
                    else:
                        running_scripts.append(Path(line))
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return running_scripts

def change_power_mode(mode : Literal['balanced' , 'power-saver' , 'performance'] , 
                      log_path : Path | None = None ,
                      vb_level : Any = 1):
    """Set Linux power profile via ``powerprofilesctl``; no-op log on Windows."""
    # running_scripts = get_running_scripts(exclude_scripts)
    main_str = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : Power set to {mode}'
    if MACHINE.is_windows:
        main_str += f' aborted due windows platform\n'
    else:
        main_str += f' applied\n'
        subprocess.run(['powerprofilesctl', 'set', mode])
    Logger.stdout(main_str , end = '')
    if log_path is not None:
        log_path.parent.mkdir(parents = True , exist_ok = True)
        with open(log_path, 'a') as log_file:
            log_file.write(main_str)

def check_process_status(pid):
    """Return ``running`` / ``sleeping`` / ``zombie`` / ``complete`` (not running) / raw status."""
    if psutil.pid_exists(pid):
        proc = psutil.Process(pid)
        status = proc.status()
        if status == psutil.STATUS_RUNNING: 
            return 'running'
        elif status == psutil.STATUS_SLEEPING: 
            return 'sleeping'
        elif status == psutil.STATUS_ZOMBIE:
            return 'zombie'
        else:
            return str(status)
    else:
        return 'complete'

def kill_process(pid):
    """Terminate then kill ``pid`` if still running; swallow errors. Returns success bool."""
    try:
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            proc.terminate()
            time.sleep(2)
            if proc.is_running():
                proc.kill()
            return True
    except Exception:
        pass
    return False
    
def argparse_dict(**kwargs):
    """Parse known args plus ``--key value`` pairs into a flat dict merged with ``kwargs``."""
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='py', help='Source of the script call')
    args , unknown = parser.parse_known_args()
    return args.__dict__ | unknown_args(unknown) | kwargs

def unknown_args(unknown):
    """Build a dict from ``parse_known_args`` tail (``--a 1 --b`` style)."""
    args = {}
    for ua in unknown:
        if ua.startswith('--'):
            key = ua[2:]
            if key not in args:
                args[key] = None
            else:
                raise ValueError(f'Duplicate argument: {key}')
        else:
            if args[key] is None:
                args[key] = ua
            elif isinstance(args[key] , tuple):
                args[key] = args[key] + (ua,)
            else:
                args[key] = (args[key] , ua)
    return args
