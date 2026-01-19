import os , psutil , subprocess , time , shlex , tempfile
from pathlib import Path
from typing import Literal

from src.proj.env import MACHINE
from src.proj.log import Logger

__all__ = ['ScriptCmd']

class ScriptCmd:
    macos_terminal_profile_name = 'Basic'
    macos_tempfile_method = False
    def __init__(self , script : str | Path , params : dict | None = None , 
                 mode: Literal['shell', 'os'] = 'shell'):
        self.script = (str(script.absolute()) if isinstance(script , Path) else script)
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
        return self.get_real_pid(self.process , self.cmd)

    def create_py_cmd(self):
        args_str = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in self.params.items() if v != ''])
        py_cmd = f'{MACHINE.python_path} {self.script} {args_str}'
        if MACHINE.is_windows:
            py_cmd = f'"{MACHINE.python_path}" "{self.script}" {args_str}'
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
            self.shell_cmd = f'start cmd /c "{self.py_cmd} & echo. & echo Task complete. Press any key to exit... & timeout /t -1 >nul"'
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
            self.process = self.run_cmd(self.os_cmd, cmd_type = 'OS')
        else:
            # run in shell
            if MACHINE.is_macos:
                if self.macos_tempfile_method:
                    self.run_in_shell_darwin_tempfile()
                else:
                    self.process = self.run_cmd(self.shell_cmd, cmd_type = 'AppleScript', input = self.apple_script_cmd)
            else:
                self.process = self.run_cmd(self.shell_cmd, cmd_type = 'Shell')
        return self

    @classmethod
    def run_cmd(cls , cmd : list[str] | str , cmd_type : str = 'Subprocess' , 
                communicating = True , input : str | None = None ,
                encoding = 'utf-8' , ):
        try:
            process = cls.Popen(cmd, encoding = encoding, communicating = communicating)
            _ , stderr = process.communicate(input = input)

            if process.returncode == 0:
                Logger.success(f"{cmd_type} command executed successfully : {cmd}")
            else:
                Logger.alert2(f"{cmd_type} command executed failed : {cmd}")
                Logger.alert2(f"Return Code: {process.returncode}")
                Logger.alert2(f"Error Outputs: {stderr}")
        except subprocess.CalledProcessError as e:
            Logger.error(f"{cmd_type} command executed failed with CalledProcessError : {cmd}")
            Logger.error(f"Return Code: {e.returncode}")
            Logger.error(f"Error Outputs: {e.stderr}")

        return process

    def run_in_shell_darwin_tempfile(self):
        assert self.mode == 'shell' , 'darwin mode does not support os mode'
        assert MACHINE.is_macos , f'Unsupported platform for run_in_shell_darwin: {MACHINE.system_name}'
        assert self.macos_tempfile_method , 'macos_tempfile_method must be True'

        with tempfile.NamedTemporaryFile(mode='w+', suffix=".applescript", delete=False) as temp_script:
            temp_script.write(self.apple_script_cmd)
            temp_script_path = Path(temp_script.name)

        try:
            self.process = self.run_cmd([*self.shell_cmd , str(temp_script_path)], cmd_type = 'AppleScript')
        finally:
            temp_script_path.unlink(missing_ok=True)

        return self

    @staticmethod
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

    @classmethod
    def get_task_id_from_cmd(cls , cmd : str):
        return cmd.split('task_id=')[1].split(' ')[0] if 'task_id=' in cmd else None

    @classmethod
    def get_real_pid(cls , process : subprocess.Popen | subprocess.CompletedProcess | None , cmd : str):
        task_id = cls.get_task_id_from_cmd(cmd)
        script_name = os.path.basename(cmd.split('.py')[0].split(' ')[-1]) + '.py'
        if MACHINE.is_linux or MACHINE.is_macos:
            return cls.find_python_process_by_name(script_name , task_id = task_id)
        elif MACHINE.is_windows:
            if (python_pid := cls.find_child_python_process_by_name(process, script_name)) is not None:
                return python_pid
            else:
                return cls.find_python_process_by_name(script_name, task_id=task_id)
        else:
            raise ValueError(f'Unsupported platform: {MACHINE.system_name}')
        
    @classmethod
    def find_child_python_process_by_name(cls , parent_process: subprocess.Popen | subprocess.CompletedProcess | None, name: str):
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

    @classmethod
    def find_python_process_by_name(cls , name : str , task_id : str  | None = None , try_times : int = 20):
        for _ in range(try_times):
            python_process = [proc for proc in psutil.process_iter(['pid', 'name', 'cmdline']) if 'python' in proc.info['name'].lower()]
            target_process = [proc for proc in python_process if name in ' '.join(proc.info['cmdline'] or []) and (task_id is None or task_id in proc.info['cmdline'])]
            if target_process: 
                return target_process[-1].info['pid']
            time.sleep(0.5)
        return 0