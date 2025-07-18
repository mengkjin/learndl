import traceback , psutil , fnmatch , platform , subprocess

from datetime import datetime
from pathlib import Path
from typing import Literal , Any
from src.basic import path as PATH

class AutoRunTask:

    def __init__(self , task_name : str , email = True , message_capturer : bool = True , 
                 source = 'not_specified' , **kwargs):
        self.task_name = task_name.replace(' ' , '_')
        self.email = email
        self.source = source
        self.kwargs = kwargs
        
        self.init_time = datetime.now()
        self.time_str = self.init_time.strftime('%Y%m%d%H%M%S')
        
        self.streamlit_files = []
        self.exit_messages = {level : [] for level in ['info' , 'warning' , 'error' , 'critical' , 'debug']}
        self.logged_messages = []
        self.error_messages = []
        from src.basic.util import CALENDAR , MessageCapturer , LogWriter , Logger , Email
        self.capturer = MessageCapturer.CreateCapturer(self.message_capturer_path if message_capturer else False , 
                                                       self.task_name , self.init_time)
        self.dprinter = LogWriter(self.log_filename)
        self.update_to = CALENDAR.update_to()

        self.emailer = Email()
        self.logger = Logger()

    def __repr__(self):
        return f'AutoRunTask(task_name = {self.task_name} , email = {self.email} , source = {self.source} , time = {self.time_str})'

    def __enter__(self):
        self.already_done = self.record_path.exists()
        # change_power_mode('balanced')
        self.dprinter.__enter__()
        self.capturer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for level , msgs in self.exit_messages.items():
            for msg in msgs:
                getattr(self.logger , level)(msg)

        if exc_type is not None:
            print(f'Error Occured! Info : ' + '-' * 20)
            print(exc_value)

            print('Traceback : ' + '-' * 20)
            print(traceback.format_exc())
            self.status = f'Error Occured! {exc_value}'
        else:
            self.status = 'Successful ' + self.task_name.replace('_' , ' ').title() + '!'
        
        self.capturer.__exit__(exc_type, exc_value, exc_traceback)
        self.dprinter.__exit__(exc_type, exc_value, exc_traceback)

        # self.emailer.attach(self.dprinter.filename)
        
        self.attach(self.emailer.ATTACHMENTS , streamlit = True , email = False)
        if not self.forfeit_task:
            self.send_email(self.task_name)
            self.record_path.touch()
        # change_power_mode('power-saver')

    def get(self , key : str , default : Any = None) -> Any:
        raw = self.kwargs.get(key , default)
        try:
            if isinstance(raw , str): raw = eval(raw)
        except:
            pass
        return raw
    
    def __getitem__(self , key : str):
        return self.kwargs[key]
        
    @property
    def record_path(self):
        return PATH.log_record.joinpath('autorun' , f'{self.task_name}.txt')
    
    @property
    def forfeit_task(self):
        return self.already_done and self.source == 'bash'
    
    @property
    def log_filename(self):
        return PATH.log_autorun.joinpath('log_txt' , f'{self.task_name}.{self.time_str}.txt')
    
    @property
    def message_capturer_path(self):
        return PATH.log_autorun.joinpath('message_capturer' , f'{self.task_name}.{self.time_str}.html')

    def today(self , format : str = '%Y%m%d'):
        return self.init_time.strftime(format)

    def send_email(self , title : str):
        if self.email or self.emailer.ATTACHMENTS: 
            title = ' '.join([*[s.capitalize() for s in self.task_name.split('_')]])
            self.emailer.send(title = title , body = self.status , confirmation_message='Autorun')

    def info(self , message : str , at_exit = False):
        if at_exit:
            self.exit_messages['info'].append(message)
        else:
            self.logger.info(message)
        self.logged_messages.append(f'INFO : {message}')
    
    def error(self , message : str , at_exit = False):
        if at_exit:
            self.exit_messages['error'].append(message)
        else:
            self.logger.error(message)
        self.logged_messages.append(f'ERROR : {message}')
        self.error_messages.append(message)

    def warning(self , message : str , at_exit = False):
        if at_exit:
            self.exit_messages['warning'].append(message)
        else:
            self.logger.warning(message)
        self.logged_messages.append(f'WARNING : {message}')

    def debug(self , message : str , at_exit = False):
        if at_exit:
            self.exit_messages['debug'].append(message)
        else:
            self.logger.debug(message)
        self.logged_messages.append(f'DEBUG : {message}')

    def critical(self , message : str , at_exit = False):
        if at_exit:
            self.exit_messages['critical'].append(message)
        else:
            self.logger.critical(message)
        self.logged_messages.append(f'CRITICAL : {message}')

    def attach(self , file : Path | str | list[Path] | list[str] , streamlit = True , email = False):
        if not isinstance(file , list): file = [Path(file)]
        if streamlit: self.streamlit_files.extend([Path(f) for f in file if f not in self.streamlit_files])
        if email: self.emailer.attach([Path(f) for f in file if f not in self.emailer.ATTACHMENTS])
        
    @property
    def final_message(self):
        return '\n'.join(self.logged_messages)
    
    
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