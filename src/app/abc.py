import os , fnmatch , platform , psutil , subprocess , time , argparse , shlex , tempfile
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
                      verbose = False):
    # running_scripts = get_running_scripts(exclude_scripts)
    main_str = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : Power set to {mode}'
    if platform.system() == 'Windows':
        main_str += f' aborted due windows platform\n'
    else:
        main_str += f' applied\n'
        subprocess.run(['powerprofilesctl', 'set', mode])
    if verbose: 
        print(main_str , end = '')
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
    
class ScriptCmd:
    apple_terminal_profile_name = 'Pro'
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
        if self.mode == 'shell':
            return str(self.shell_cmd)
        elif self.mode == 'os':
            return str(self.py_cmd)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
        
    def run(self):
        if self.mode == 'os':
            self.run_in_os()
        elif platform.system() == 'Linux' and os.name == 'posix':
            self.run_in_linux()
        elif platform.system() == 'Windows':
            self.run_in_windows()
        elif platform.system() == 'Darwin':
            self.run_in_darwin()
        else:
            raise ValueError(f'Unsupported platform: {platform.system()}')
        return self
    
    @property
    def real_pid(self):
        return get_real_pid(self.process , self.cmd)

    def create_py_cmd(self):
        args_str = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in self.params.items() if v != ''])
        py_cmd = f'{MACHINE.python_path} {self.script} {args_str}'
        if platform.system() == 'Windows':
            py_cmd = f'{MACHINE.python_path} -c {self.script} {args_str}'
            py_cmd = py_cmd.replace("'", "'\"'\"'")
        else:
            py_cmd = f'{MACHINE.python_path} {self.script} {args_str}'
        self.py_cmd = py_cmd

    def create_os_cmd(self):
        self.os_cmd = shlex.split(self.py_cmd)

    def create_shell_cmd(self):
        if platform.system() == 'Linux' and os.name == 'posix':
            self.shell_cmd = ['gnome-terminal' , '--' , 'bash' , '-c' , f'{self.py_cmd}; echo "Task complete. Press any key to exit..."; read -n 1 -s']
            # self.shell_cmd = f'gnome-terminal -- bash -c "{self.py_cmd}; exec bash; exit"'
        elif platform.system() == 'Windows':
            self.shell_cmd = f'start cmd /c {self.py_cmd} && pause'
        elif platform.system() == 'Darwin':
            self.shell_cmd = f'''
tell application "Terminal"
    -- Create a new terminal window
    set new_window to do script ""
    set current settings of new_window to settings set "{self.apple_terminal_profile_name}"
    do script "{self.py_cmd}; echo 'Task complete. Press any key to exit...'; read -k 1; exit" in new_window
    -- Bring the main application window to the front
    activate
end tell
'''
        else:
            raise ValueError(f'Unsupported platform: {platform.system()}')

    def run_in_os(self):
        assert not self.shell , 'os mode does not support shell mode'
        self.process = subprocess.Popen(self.os_cmd, shell=False , encoding='utf-8')

    def run_in_linux(self):
        assert self.shell , 'linux mode does not support os mode'
        assert platform.system() == 'Linux' and os.name == 'posix' , f'Unsupported platform for run_in_linux: {platform.system()}'
        self.process = subprocess.Popen(self.shell_cmd, shell=True , encoding='utf-8')
        return self
    
    def run_in_windows(self):
        assert self.shell , 'windows mode does not support os mode'
        assert platform.system() == 'Windows' , f'Unsupported platform for run_in_windows: {platform.system()}'
        self.process = subprocess.Popen(self.shell_cmd, shell=True , encoding='utf-8')
        return self

    def run_in_darwin(self):
        assert isinstance(self.shell_cmd , str) , 'shell_cmd must be a string in darwin mode'
        assert platform.system() == 'Darwin' , f'Unsupported platform for run_in_darwin: {platform.system()}'
        try:
            self.process = subprocess.Popen(
                ["osascript", "-"], 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True # For string input/output
            )
            _ , stderr = self.process.communicate(input=self.shell_cmd)

            if self.process.returncode == 0:
                Logger.success("AppleScript executed successfully.")
            else:
                Logger.error(f"AppleScript failed with return code {self.process.returncode}")
                Logger.error(f"Errors: {stderr}")
        except subprocess.CalledProcessError as e:
            Logger.error(f"AppleScript failed with error:\n{e.stderr}")
        except Exception as e:
            Logger.error(f"An unexpected error occurred: {e}")

    def run_in_darwin_tempfile(self):
        assert isinstance(self.shell_cmd , str) , 'shell_cmd must be a string in darwin mode'
        assert platform.system() == 'Darwin' , f'Unsupported platform for run_in_darwin_tempfile: {platform.system()}'
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".applescript", delete=False) as temp_script:
            temp_script.write(self.shell_cmd)
            temp_script_path = temp_script.name

        try:
            self.process = subprocess.Popen(
                ["osascript", temp_script_path],
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True # For string input/output
            )
            _ , stderr = self.process.communicate()

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
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)

def get_task_id_from_cmd(cmd : str):
    return cmd.split('task_id=')[1].split(' ')[0] if 'task_id=' in cmd else None

def get_real_pid(process : subprocess.Popen | subprocess.CompletedProcess | None , cmd : str):
    task_id = get_task_id_from_cmd(cmd)
    script_name = os.path.basename(cmd.split('.py')[0].split(' ')[-1]) + '.py'
    if (platform.system() == 'Linux' and os.name == 'posix') or (platform.system() == 'Darwin'):
        return find_python_process_by_name(script_name , task_id = task_id)
    elif platform.system() == 'Windows':
        if (python_pid := find_child_python_process_by_name(process, script_name)) is not None:
            return python_pid
        else:
            return find_python_process_by_name(script_name, task_id=task_id)
    else:
        raise ValueError(f'Unsupported platform: {platform.system()}')
    
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
