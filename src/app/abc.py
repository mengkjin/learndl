import os , fnmatch , platform , psutil , subprocess , time , argparse , shlex
from datetime import datetime
from pathlib import Path
from typing import Literal

from src.project_setting import MACHINE

DEFAULT_EXCLUDES = ['kernel_interrupt_daemon.py']

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
    
class ScriptCmd:
    def __init__(self , script : str | Path , params : dict | None = None , mode: Literal['shell', 'os'] = 'shell'):
        self.script = str(script.absolute()) if isinstance(script , Path) else script
        self.params = params or {}
        assert mode in ['shell', 'os'] , f'Invalid mode: {mode}'
        self.mode = mode

        args_str = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in self.params.items() if v != ''])
        py_cmd = f'{MACHINE.python_path} {self.script} {args_str}'
        self.py_cmd = py_cmd
        self.os_cmd = shlex.split(py_cmd)
        if platform.system() == 'Linux' and os.name == 'posix':
            cmd = f'gnome-terminal -- bash -c "{py_cmd}; exec bash"'
        elif platform.system() == 'Windows':
            cmd = f'start cmd /k {py_cmd}'
        elif platform.system() == 'Darwin':
            cmd = f'''osascript -e 'tell application "Terminal" to do script "{py_cmd}; exec bash"' '''
        else:
            raise ValueError(f'Unsupported platform: {platform.system()}')
        self.shell_cmd = cmd

    def __repr__(self):
        return f'TerminalCmd(script={self.script}, params={self.params}, mode={self.mode})'

    @property
    def cmd(self):
        return str(self)
    
    @property
    def shell(self):
        return self.mode == 'shell'
    
    @property
    def use_cmd(self):
        if self.mode == 'shell':
            return self.shell_cmd
        elif self.mode == 'os':
            return self.os_cmd
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
        
    def __str__(self):
        if self.mode == 'shell':
            return self.shell_cmd
        elif self.mode == 'os':
            return self.py_cmd
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
        
    def run(self):
        self.process = subprocess.Popen(self.use_cmd, shell=self.shell , encoding='utf-8')
        return self
    
    @property
    def pid(self):
        return self.process.pid
    
    @property
    def real_pid(self):
        return get_real_pid(self.process , self.cmd)

def terminal_cmd_old(script : str | Path , params : dict | None = None , close_after_run = False ,
                mode: Literal['shell', 'os'] = 'shell'):
    params = params or {}
    if isinstance(script , Path): script = str(script.absolute())
    args = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in params.items() if v != ''])
    cmd = f'{MACHINE.python_path} {script} {args}'
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

def get_task_id_from_cmd(cmd : str):
    return cmd.split('task_id=')[1].split(' ')[0] if 'task_id=' in cmd else None

def get_real_pid(process : subprocess.Popen , cmd : str):
    task_id = get_task_id_from_cmd(cmd)
    script_name = os.path.basename(cmd.split('.py')[0].split(' ')[-1]) + '.py'
    if (platform.system() == 'Linux' and os.name == 'posix') or (platform.system() == 'Darwin'):
        return find_python_process_by_name(script_name , task_id = task_id)
    elif platform.system() == 'Windows':
        if (python_pid := find_child_python_process_by_name(process.pid, script_name)) is not None:
            return python_pid
        else:
            return find_python_process_by_name(script_name, task_id=task_id)
    else:
        raise ValueError(f'Unsupported platform: {platform.system()}')
    
def find_child_python_process_by_name(parent_pid: int, name: str):
    """find child python process in Windows"""
    try:
        children = psutil.Process(parent_pid).children(recursive=True)
        for child in children:
            if 'python' in child.name().lower() and name in ' '.join(child.cmdline()):
                return child.pid
        return None
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def find_python_process_by_name(name : str , task_id : str  | None = None , try_times : int = 20):
    for _ in range(try_times):
        python_process = [proc for proc in psutil.process_iter(['pid', 'name', 'cmdline']) if 'python' in proc.info['name'].lower()]
        target_process = [proc for proc in python_process if name in ' '.join(proc.info['cmdline'] or []) and (task_id is None or task_id in proc.info['cmdline'])]
        if target_process: 
            return target_process[-1].info['pid']
        time.sleep(0.5)
    raise ValueError(f'No python process found for name: {name}')
    
def argparse_dict(**kwargs):
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='py', help='Source of the script call')
    parser.add_argument('--email', type=str, default='0', help='Send email or not')
    args , unknown = parser.parse_known_args()
    return args.__dict__ | unknown_args(unknown) | kwargs

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
