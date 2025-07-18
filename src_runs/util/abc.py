import os , fnmatch , platform , psutil , subprocess , time , argparse , traceback
from dataclasses import dataclass , field
from datetime import datetime
from pathlib import Path
from typing import Literal , Callable , Any

from src_runs.util import update_exit_message

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
        if target_process: 
            return target_process[-1].info['pid']
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

class BackendTaskManager:
    '''
    convert script main function to one that can be used as a task in streamlit project
    example:
        @BackendTaskManager.manage(x = 1)
        def test(x : int , **kwargs):
            return 'yes' , [Path('test.txt') , Path('test.csv')]
            # return BackendTaskManager.ExitMessage(message = 'yes' , files = ['test.txt' , 'test.csv'])
            # return {'message' : 'yes' , 'files' : ['test.txt' , 'test.csv']}
    manage params:
        will passed to the task as kwargs
    return:
        will be used as exit message of the task
        - None
        - ExitMessage 
        - str
        - Path
        - tuple of 2 elements (message , list of files)
        - tuple of n elements (str as message , Path as file)
        - list of str (as message) or Path (as file)
        - dict of key-value pairs (message , files , code , error)
        - AutoRunTask object
        - any other type (converted to str)
    '''
    def __init__(self , **kwargs):
        self.params = argparse_dict(**kwargs)
        self.exit_msg = {
            'pid': os.getpid(),
            'end_time': None,
            'status': None,
            'exit_code': None,
            'exit_error': None,
            'exit_message': None,
            'exit_files': None,
        }

    def __enter__(self):
        self.task_id = self.params.get('task_id')
        return self

    def __exit__(self , exc_type , exc_value , exc_traceback):
        self.exit_msg['end_time'] = time.time()
        if exc_type is None:
            self.exit_msg['status'] = 'error' if self.exit_msg['exit_code'] else 'complete'
        else:
            self.exit_msg['status'] = 'error'
            self.exit_msg['exit_code'] = 1
            self.exit_msg['exit_message'] = str(exc_value)
            self.exit_msg['exit_error'] = traceback.format_exc()
        update_exit_message(self.task_id, **self.exit_msg)

    @dataclass(slots = True)
    class ExitMessage:
        '''
        can use ExitMessage.from_return(ret) to convert return to ExitMessage
        ret can be of variaous form
        '''
        message : str | None = None
        files : list[Any] | None = None
        code : int = 0
        error : str | None = None

        @classmethod
        def from_return(cls , ret : Any):
            if isinstance(ret , cls):
                return ret
            elif isinstance(ret , str):
                return cls(message = ret)
            elif isinstance(ret , Path):
                return cls(files = [str(ret)])
            elif isinstance(ret , tuple):
                if len(ret) == 2 and isinstance(ret[0] , str) and isinstance(ret[1] , list):
                    return cls(message = ret[0] , files = [str(x) for x in ret[1]])
                else:
                    message = []
                    files = []
                    for x in ret:
                        if isinstance(x , Path):
                            files.append(str(x))
                        else:
                            message.append(str(x))
                    return cls(message = '\n'.join(message) , files = files)
            elif isinstance(ret , list):
                if not ret:
                    return cls()
                elif isinstance(ret[0] , Path):
                    return cls(files = [str(x) for x in ret])
                else:
                    return cls(message = '\n'.join(ret))
            elif isinstance(ret , dict):
                return cls(**{k:v for k,v in ret.items() if k in cls.__slots__})
            elif ret.__class__.__name__ == 'AutoRunTask':
                return cls(message = ret.final_message , files = ret.streamlit_files , 
                           code = len(ret.error_messages) , error = '\n'.join(ret.error_messages))
            else:
                return cls(message = str(ret))

    def exit_message(self , exit_msg : ExitMessage):
        if not self.task_id: return
        if exit_msg.message: self.exit_msg['exit_message'] = exit_msg.message
        if exit_msg.files:   self.exit_msg['exit_files'] = [str(f) for f in exit_msg.files]
        if exit_msg.code:    self.exit_msg['exit_code'] = exit_msg.code
        if exit_msg.error:   self.exit_msg['exit_error'] = exit_msg.error
        
    @classmethod
    def manage(cls , **params):
        '''
        use BackendTaskManager to manage task, take acceptable return as exit message
        params will be passed to the warpped function as kwargs
        example:
        @BackendTaskManager.manage(x = 1)
        def test(x : int , **kwargs):
            return 'yes' , Path('test.txt') , Path('test.csv')
            # return BackendTaskManager.ExitMessage(message = 'yes' , files = ['test.txt' , 'test.csv'])
            # return {'message' : 'yes' , 'files' : ['test.txt' , 'test.csv']}
        '''
        def inner(func : Callable):
            def wrapper(*args , **kwargs):
                with cls(**params) as bm:
                    ret = func(*args , **kwargs , **bm.params)
                    bm.exit_message(cls.ExitMessage.from_return(ret))
                return ret
            return wrapper
        return inner
