"""Process discovery, power profile (non-Windows), and CLI arg parsing helpers."""

import fnmatch , psutil , subprocess , time , argparse
from datetime import datetime
from pathlib import Path
from typing import Any , Literal

from src.proj.env import MACHINE
from src.proj.log import Logger

__all__ = [
    'get_running_scripts' , 'change_power_mode' , 'check_process_status' , 
    'kill_process' , 'argparse_dict' , 'unknown_args' , 
    'ask_for_confirmation' , 'ask_for_selections' , 'ask_for_retry']

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
        with open(log_path, 'a' , encoding='utf-8') as log_file:
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

class AskFlag:
    """
    Ask for confirmation, selections, or retry.
    
    Args:
        flag : Literal['yes' , 'no' , 'abort']
        result : Any = None
        
    Returns:
        AskFlag object with the following properties:
        - yes : bool
        - no : bool
        - abort : bool
        - result : Any
          - list of selections for ask_for_selections
    """
    def __init__(
        self , 
        flag : Literal['yes' , 'no' , 'abort'] ,
        result : Any = None):
        self.flag : Literal['yes' , 'no' , 'abort'] = flag
        self.result : Any = result

    def __repr__(self) -> str:
        return f'AskFlag({self.flag})'
    def __str__(self) -> str:
        return self.flag

    @property
    def yes(self) -> bool:
        return self.flag == 'yes'
    @property
    def no(self) -> bool:
        return self.flag == 'no'
    @property
    def abort(self) -> bool:
        return self.flag == 'abort'

    def __bool__(self) -> bool:
        return self.yes

def ask_for_confirmation(msg = '' , timeout = -1 , ask_times = 1):
    """Prompt up to ``recurrent`` times with optional per-prompt timeout.

    Returns:
        Tuple of (inputs list, bool list from ``proceed_condition``).
    """
    
    from pytimedinput import timedInput
    for i in range(ask_times):
        prefix = f'{msg} Please confirm (y/n/q) ({i+1}/{ask_times} rounds): '
            
        value, is_timeout = None , False
        if timeout > 0:
            try:
                value, is_timeout = timedInput(f'{prefix} (in {timeout} seconds): ' , timeout = timeout)
            except Exception:
                pass
        if value is None : 
            value, is_timeout = input(f'{prefix} : ') , False
        if value.lower() not in ['y' , 'n' , 'q']:
            Logger.error(f'Invalid input: {value}')
            return AskFlag('no')
        if is_timeout:
            Logger.stdout(f'Input is timed out at the {i+1}th round.')
            return AskFlag('no')
        elif value.lower() in ['n' , 'q']:
            Logger.stdout(f'Confirmation is rejected at the {i+1}th round.')
            return AskFlag('no')
        
    return AskFlag('yes')

def ask_for_selections(msg : str , options : int , start : int = 1) -> AskFlag:
    """Parse the selections."""
    min , max = start , options + start - 1
    
    selection = input(f'{msg} ({min}-{max}, seperated by comma , q to quit): ')
    if selection.lower() == 'q':
        return AskFlag('no')
    selections = [s.strip() for s in selection.split(',') if s.strip()]
    if any(not s.isdigit() for s in selections):
        Logger.error(f'Contains non-digit characters: {selection}')
        return AskFlag('abort')

    selections = [int(i) for i in selections]
    if any(s < start or s > options + start - 1 for s in selections):
        Logger.error(f'Contains indices out of range [{min}-{max}]: {selection}')
        return AskFlag('abort')

    flag = input(f'Are you sure to select {selections}? (press y to confirm): ')
    if flag.lower() == 'y':
        return AskFlag('yes' , result = selections)
    else:
        return AskFlag('abort')

def ask_for_retry(msg : str) -> AskFlag:
    """Ask for exit."""
    while True:
        flag = input(f'{msg} (y/n/q): ')
        if flag.lower() in ['n' , 'q']:
            return AskFlag('no')
        elif flag.lower() == 'y':
            return AskFlag('yes')
        else:
            Logger.error(f'Invalid input: {flag}')
