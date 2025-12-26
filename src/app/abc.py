import os , fnmatch , psutil , subprocess , time , argparse , shlex , tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from src.proj import MACHINE , Logger

DEFAULT_EXCLUDES = ['kernel_interrupt_daemon.py']

def get_running_scripts(exclude_scripts : list[str] | str | None = None , script_type = ['*.py']):
    running_scripts : list[Path] = []
    if isinstance(exclude_scripts , str): 
        exclude_scripts = [exclude_scripts]
    excludes = [Path(scp).name for scp in (exclude_scripts or []) + DEFAULT_EXCLUDES]
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
                      vb_level : int = 1):
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
    except Exception:
        pass
    return False

def Popen(cmd : list[str] | str , encoding = 'utf-8' , communicating = False):
    kwargs = {
        'shell' : isinstance(cmd , str) ,
        'encoding' : encoding , 
        'stdin': subprocess.PIPE if communicating else None,
        'stdout': subprocess.PIPE if communicating else None,
        'stderr': subprocess.PIPE if communicating else None,
        'text': True if communicating else None,
    }
    return subprocess.Popen(cmd , **kwargs)
    
class ScriptCmd:
    macos_terminal_profile_name = 'Basic'
    macos_tempfile_method = False
    def __init__(self , script : str | Path , params : dict | None = None , 
                 mode: Literal['shell', 'os'] = 'shell'):
        self.script = str(script.absolute()) if isinstance(script , Path) else script
        self.params = params or {}
        assert mode in ['shell', 'os'] , f'Invalid mode: {mode}'
        self.mode = mode

        self.create_py_cmd()
        self.create_os_cmd()
        self.create_shell_cmd()

    def __repr__(self):
        return f'TerminalCmd(script={self.script}, params={self.params}, mode={self.mode})'

    @property
    def cmd(self):
        return str(self)
    
    @property
    def shell(self):
        return self.mode == 'shell'

    def __str__(self):
        if self.mode == 'os':
            return str(self.py_cmd)
        elif self.mode == 'shell':
            if MACHINE.is_macos:
                return str([*self.shell_cmd , self.apple_script_cmd])
            else:
                return str(self.shell_cmd)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
    
    @property
    def real_pid(self):
        return get_real_pid(self.process , self.cmd)

    def create_py_cmd(self):
        args_str = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in self.params.items() if v != ''])
        py_cmd = f'{MACHINE.python_path} {self.script} {args_str}'
        if MACHINE.is_windows:
            py_cmd = f'"{MACHINE.python_path}" -c "{self.script} {args_str}"'
            py_cmd = py_cmd.replace("'" , "\\'").replace('"' , '\\"')
        else:
            py_cmd = f'{MACHINE.python_path} {self.script} {args_str}'
        self.py_cmd = py_cmd

    def create_os_cmd(self):
        self.os_cmd = shlex.split(self.py_cmd)

    def create_shell_cmd(self):
        if MACHINE.is_linux:
            self.shell_cmd = ['gnome-terminal' , '--' , 'bash' , '-c' , f'{self.py_cmd}; echo \'Task complete. Press any key to exit...\'; read -n 1 -s']
            # self.shell_cmd = f'gnome-terminal -- bash -c "{self.py_cmd}; echo \'Task complete. Press any key to exit...\'; read -n 1 -s"'
            # self.shell_cmd = f'gnome-terminal -- bash -c "{self.py_cmd}; exec bash; exit"'
        elif MACHINE.is_windows:
            # self.shell_cmd = f'start cmd /c "{self.py_cmd} && echo. && echo "Task complete. Press any key to exit..." && pause >nul"'
            self.shell_cmd = f'start cmd /c "{self.py_cmd} && echo. && echo "Task complete. Press any key to exit..." && timeout /t -1 >nul"'
        elif MACHINE.is_macos:
            self.shell_cmd = ["osascript"] if self.macos_tempfile_method else ["osascript", "-"]
            self.apple_script_cmd = f'''
tell application "Terminal"
    -- Create a new terminal window
    set new_window to do script ""
    set current settings of new_window to settings set "{self.macos_terminal_profile_name}"
    do script "{self.py_cmd}; echo 'Task complete. Press any key to exit...'; read -k 1; exit" in new_window
    -- Bring the main application window to the front
    activate
end tell
'''
        else:
            raise ValueError(f'Unsupported platform: {MACHINE.system_name}')

    def run(self):
        if self.mode == 'os':
            self.process = Popen(self.os_cmd)
        elif MACHINE.is_linux or MACHINE.is_windows:
            self.process = Popen(self.shell_cmd)
        elif MACHINE.is_macos:
            self.run_in_darwin()
        else:
            raise ValueError(f'Unsupported platform: {MACHINE.system_name}')
        return self

    def run_in_darwin(self):
        assert self.mode == 'shell' , 'darwin mode does not support os mode'
        assert MACHINE.is_macos , f'Unsupported platform for run_in_darwin: {MACHINE.system_name}'

        if self.macos_tempfile_method:
            with tempfile.NamedTemporaryFile(mode='w+', suffix=".applescript", delete=False) as temp_script:
                temp_script.write(self.apple_script_cmd)
                temp_script_path = Path(temp_script.name)

        try:
            if self.macos_tempfile_method:
                self.process = Popen([*self.shell_cmd , str(temp_script_path)], communicating = True)
                _ , stderr = self.process.communicate()
            else:
                self.process = Popen(self.shell_cmd, communicating = True)
                _ , stderr = self.process.communicate(input=self.apple_script_cmd)

            if self.process.returncode == 0:
                Logger.success("AppleScript executed successfully.")
            else:
                Logger.error(f"AppleScript failed with return code {self.process.returncode}")
                Logger.error(f"Errors: {stderr}")
        except subprocess.CalledProcessError as e:
            Logger.error(f"AppleScript failed with error:\n{e.stderr}")
        except Exception as e:
            Logger.error(f"An unexpected error occurred: {e}")
        finally:
            if self.macos_tempfile_method:
                temp_script_path.unlink(True)

def get_task_id_from_cmd(cmd : str):
    return cmd.split('task_id=')[1].split(' ')[0] if 'task_id=' in cmd else None

def get_real_pid(process : subprocess.Popen | subprocess.CompletedProcess | None , cmd : str):
    task_id = get_task_id_from_cmd(cmd)
    script_name = os.path.basename(cmd.split('.py')[0].split(' ')[-1]) + '.py'
    if MACHINE.is_linux or MACHINE.is_macos:
        return find_python_process_by_name(script_name , task_id = task_id)
    elif MACHINE.is_windows:
        if (python_pid := find_child_python_process_by_name(process, script_name)) is not None:
            return python_pid
        else:
            return find_python_process_by_name(script_name, task_id=task_id)
    else:
        raise ValueError(f'Unsupported platform: {MACHINE.system_name}')
    
def find_child_python_process_by_name(parent_process: subprocess.Popen | subprocess.CompletedProcess | None, name: str):
    """find child python process in Windows"""
    try:
        if parent_process is None or isinstance(parent_process, subprocess.CompletedProcess):
            return None
        children = psutil.Process(parent_process.pid).children(recursive=True)
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
    return 0
    
def argparse_dict(**kwargs):
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='py', help='Source of the script call')
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
