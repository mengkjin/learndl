import traceback , psutil , fnmatch , platform , subprocess

from datetime import datetime
from pathlib import Path
from typing import Literal , Any
from src.basic import path as PATH

PATH.log_record.joinpath('autorun').mkdir(parents = True , exist_ok = True)

class AutoRunTask:
    def __init__(self , task_name : str , email = True , email_if_attachment = True , 
                 source = 'not_specified' , message_capturer : Path | str | bool = False , **kwargs):
        self.task_name = task_name.replace(' ' , '_')
        self.email = email
        self.email_if_attachment = email_if_attachment
        self.source = source

        from src.basic.util.email import Email
        self.emailer = Email()

        from src.basic.util.logger import MessageCapturer , DualPrinter
        self.capturer = MessageCapturer.CreateCapturer(message_capturer)
        self.dprinter = DualPrinter()

    def __enter__(self):
        self.already_done = self.record_path.exists()
        # change_power_mode('balanced')
        self.time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dprinter.set_attrs(f'{self.task_name}/{self.task_name}.{self.time_str}.txt')
        self.dprinter.__enter__()
        self.capturer.set_attrs(f'{self.task_name}')
        self.capturer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            print(f'Error Occured! Info : ' + '-' * 20)
            print(exc_value)

            print('Traceback : ' + '-' * 20)
            print(traceback.format_exc())
            self.status = f'Error Occured! {exc_value}'
        else:
            self.status = 'Successful ' + self.task_name.replace('_' , ' ').title() + '!'
        
        self.dprinter.__exit__(exc_type, exc_value, exc_traceback)
        self.capturer.__exit__(exc_type, exc_value, exc_traceback)
        
        if self.forfeit_task: return
        self.send_email(self.task_name)
        # change_power_mode('power-saver')

        self.record_path.touch()

    @property
    def record_path(self):
        return PATH.log_record.joinpath('autorun' , f'{self.task_name}.txt')
    
    @property
    def forfeit_task(self):
        return self.already_done and self.source == 'bash'

    def send_email(self , title : str):
        if self.email or (self.email_if_attachment and self.emailer.ATTACHMENTS): 
            title = ' '.join([*[s.capitalize() for s in self.task_name.split('_')]])
            self.emailer.attach(self.dprinter.filename)
            self.emailer.send(title = title , body = self.status , confirmation_message='Autorun')

def get_running_scripts(exclude_scripts : list[str] | str | None = None , script_type = ['*.py']):
    running_scripts : list[Path] = []
    if isinstance(exclude_scripts , str): exclude_scripts = [exclude_scripts]
    excludes = [Path(scp).name for scp in (exclude_scripts or []) + ['kernel_interrupt_daemon.py']]
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