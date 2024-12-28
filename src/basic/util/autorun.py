import traceback , psutil , fnmatch , platform , subprocess

from datetime import datetime
from pathlib import Path
from typing import Literal
from src.basic import path as PATH
from .logger import DualPrinter
from .email import Email

class AutoRunTask:
    def __init__(self , task_name : str , email = True , email_if_attachment = True ,**kwargs):
        self.task_name = task_name.replace(' ' , '_')
        self.email = email
        self.email_if_attachment = email_if_attachment

    def __enter__(self):
        # change_power_mode('balanced')
        self.date_str = datetime.now().strftime('%Y%m%d')
        self.time_str = datetime.now().strftime('%H%M%S')
        name = '.'.join([self.task_name , f'{self.date_str}_{self.time_str}' , 'txt'])
        self.printer = DualPrinter(f'{self.task_name}/{name}')
        self.printer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            print(f'Error Occured! Info : ' + '-' * 20)
            print(exc_value)

            print('Traceback : ' + '-' * 20)
            print(traceback.format_exc())
            self.status = f'Error Occured! {exc_value}'
        else:
            self.status = ' '.join(['Successful' , *[s.capitalize() for s in self.task_name.split('_')]]) + '!'
        self.printer.__exit__(exc_type, exc_value, exc_traceback)
        if self.email or (self.email_if_attachment and Email.ATTACHMENTS): 
            title = ' '.join([*[s.capitalize() for s in self.task_name.split('_')] , 'at' , self.date_str])
            Email.attach(self.printer.filename)
            Email().send(title = title , body = self.status)
        # change_power_mode('power-saver')

def change_power_mode(mode : Literal['balanced' , 'power-saver' , 'performance'] , 
                      log_path : Path | None = PATH.logs.joinpath('suspend','power_check.log') ,
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
        

def get_running_scripts(exclude_scripts : list[str] = [] , script_type = ['*.py'] , default_excludes = ['kernel_interrupt_daemon.py']):
    running_scripts : list[Path] = []
    self = Path(__file__)
    exclude_scripts = exclude_scripts + default_excludes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline: continue
            for line in cmdline:
                if any(fnmatch.fnmatch(line, pattern) for pattern in script_type):
                    script = Path(line)
                    if script == self or any(scp in line for scp in exclude_scripts): 
                        pass
                    else:
                        running_scripts.append(Path(line))
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return running_scripts

def suspend_this_machine(log_path : Path | None = PATH.logs.joinpath('suspend','suspend.log') , verbose = False):
    main_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if platform.system() == 'Windows':
        main_str += f' : Suspension aborted due windows platform\n'
    else:
        main_str += f' : Suspension applied\n'
    if verbose: print(main_str , end = '')
    if log_path is not None:
        log_path.parent.mkdir(parents = True , exist_ok = True)
        with open(log_path, 'a') as log_file:
            log_file.write(main_str)
    if platform.system() != 'Windows':
        subprocess.run(['systemctl', 'suspend'])