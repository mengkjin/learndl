import os , fnmatch , platform , psutil , subprocess , time , argparse
from datetime import datetime
from pathlib import Path
from typing import Literal

DEFAULT_EXCLUDES = ['kernel_interrupt_daemon.py']

def edit_file(file_path : Path | str):
    path = Path(file_path).absolute()
    if os.name == 'nt':  # Windows
        editor_command = f'notepad {str(path)}'
    elif os.name == 'posix':  # Linux
        editor_command = f'nano {str(path)}'
    process = subprocess.Popen(editor_command, shell=True)
    process.communicate()

def get_running_scripts(exclude_scripts : list[str] | str | None = None , script_type = ['*.py']):
    running_scripts : list[Path] = []
    if isinstance(exclude_scripts , str): exclude_scripts = [exclude_scripts]
    excludes = [Path(scp).name for scp in (exclude_scripts or []) + DEFAULT_EXCLUDES]
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline: continue
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
                      verbose = False):
    # running_scripts = get_running_scripts(exclude_scripts)
    main_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f' : Power set to {mode}'
    if platform.system() == 'Windows':
        main_str += f' aborted due windows platform\n'
    else:
        main_str += f' applied\n'
        subprocess.run(['powerprofilesctl', 'set', mode])
    if verbose: print(main_str , end = '')
    if log_path is not None:
        log_path.parent.mkdir(parents = True , exist_ok = True)
        with open(log_path, 'a') as log_file:
            log_file.write(main_str)


def check_process_status(pid):
    """check process status"""
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
    """kill process"""
    try:
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            proc.terminate()
            time.sleep(2)
            if proc.is_running():
                proc.kill()
            return True
    except:
        pass
    return False

def python_path():
    if platform.system() == 'Linux' and os.name == 'posix':
        return 'python3.10'
    elif platform.system() == 'Darwin':
        return 'source /Users/mengkjin/workspace/learndl/.venv/bin/activate; python'
    else:
        return 'python'

def terminal_cmd(script : str | Path , params : dict | None = None , close_after_run = False):
    params = params or {}
    if isinstance(script , Path): script = str(script.absolute())
    args = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in params.items() if v != ''])
    cmd = f'{python_path()} {script} {args}'
    if platform.system() == 'Linux' and os.name == 'posix':
        if not close_after_run: cmd += '; exec bash'
        cmd = f'gnome-terminal -- bash -c "{cmd}"'
    elif platform.system() == 'Windows':
        # cmd = f'start cmd /k {cmd}'
        if not close_after_run: 
            cmd = f'start cmd /k {cmd}'
        pass
    elif platform.system() == 'Darwin':
        if not close_after_run:
            cmd += '; exec bash'
        else:
            cmd += '; exit'
        cmd = f'''osascript -e 'tell application "Terminal" to do script "{cmd}"' '''
    else:
        raise ValueError(f'Unsupported platform: {platform.system()}')
    return cmd

def get_real_pid(process : subprocess.Popen , cmd : str):
    if platform.system() == 'Linux' and os.name == 'posix':
        return process.pid
    elif platform.system() == 'Windows':
        return process.pid
    elif platform.system() == 'Darwin':
        name = cmd.split('.py')[0].split('/')[-1] + '.py'
        if 'task_id=' in cmd:
            task_id = cmd.split('task_id=')[1].split(' ')[0]
        else:
            task_id = None
        return find_python_process_by_name(name , task_id = task_id)
    else:
        raise ValueError(f'Unsupported platform: {platform.system()}')
    
def find_python_process_by_name(name : str , task_id : str  | None = None , try_times : int = 20):
    for _ in range(try_times):
        python_process = [proc for proc in psutil.process_iter(['pid', 'name', 'cmdline']) if 'python' in proc.info['name'].lower()]
        target_process = [proc for proc in python_process if name in ' '.join(proc.info['cmdline'] or []) and (task_id is None or task_id in proc.info['cmdline'])]
        if target_process: return target_process[-1].info['pid']
        time.sleep(0.5)
    raise ValueError(f'No python process found for name: {name}')
    
def argparse_dict(**kwargs):
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='', help='Source of the script call')
    parser.add_argument('--email', type=int, default=0, help='Send email or not')
    args , unknown = parser.parse_known_args()
    return kwargs | args.__dict__ | unknown_args(unknown)

def unknown_args(unknown):
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
