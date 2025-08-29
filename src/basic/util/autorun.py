import traceback , psutil , fnmatch , platform , subprocess

from datetime import datetime
from pathlib import Path
from typing import Literal , Any

from src.proj import MACHINE , PATH

class AutoRunTask:
    def __init__(self , task_name : str , task_key : str | Any | None = None , email = True , message_capturer : bool = True , source = 'py' , **kwargs):
        self.task_name = task_name.replace(' ' , '_').lower()
        self.task_key = str(task_key).lower() if task_key is not None else None
        self.task_full_name = task_name if task_key is None else f'{task_name}_{task_key}'
        self.email = email
        self.kwargs = kwargs
        self.source = source
        self.init_time = datetime.now()
        self.time_str = self.init_time.strftime('%Y%m%d%H%M%S')
        
        self.exit_files = []
        self.exit_messages = {level : [] for level in ['info' , 'warning' , 'error' , 'critical' , 'debug']}
        self.logged_messages = []
        self.error_messages = []

        import src.basic.util as U
        self.capturer = U.MessageCapturer.CreateCapturer(self.message_capturer_path if message_capturer else False , 
                                                         self.task_full_name , self.init_time)
        self.dprinter = U.LogWriter(self.log_filename)
        self.update_to = U.CALENDAR.update_to()
        self.task_recorder = U.TaskRecorder('autorun' , self.task_name , self.task_key or '')
        self.emailer = U.Email()
        self.logger = U.Logger()

        self.status = 'Starting'

    def __repr__(self):
        return f'AutoRunTask(task_name = {self.task_full_name} , email = {self.email} , source = {self.source} , time = {self.time_str})'

    def __enter__(self):
        self.already_done = self.task_recorder.is_finished()
        # change_power_mode('balanced')
        self.dprinter.__enter__()
        self.capturer.__enter__()
        self.status = 'Running'
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = datetime.now()
        for level , msgs in self.exit_messages.items():
            for msg in msgs:
                getattr(self.logger , level)(msg)
        if exc_type is not None:
            # print(f'Error Occured! Info : ' + '-' * 20)
            # print(exc_value)

            # print('Traceback : ' + '-' * 20)
            # print(traceback.format_exc())
            self.status = 'Error'
            self.error_messages.append('\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        else:
            self.status = 'Success'
        
        self.capturer.__exit__(exc_type, exc_value, exc_traceback)
        self.dprinter.__exit__(exc_type, exc_value, exc_traceback)

        self.attach(self.emailer.Attachments.get('default' , []) , streamlit = True , email = False)
        if not self.forfeit_task:
            self.send_email()
            if self.status == 'Success': 
                self.task_recorder.mark_finished(
                    remark = ' | '.join([f'source: {self.source}' , 
                                       f'exit_code: {len(self.error_messages)}']))
        # change_power_mode('power-saver')

    @property
    def success(self):
        return self.status == 'Success'

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
    def forfeit_task(self):
        return self.already_done and self.source == 'bash'
    
    @property
    def log_filename(self):
        return PATH.log_autorun.joinpath('log_txt' , f'{self.task_full_name}.{self.time_str}.txt')
    
    @property
    def message_capturer_path(self):
        return PATH.log_autorun.joinpath('message_capturer' , f'{self.task_full_name}.{self.time_str}.html')
    
    @property
    def execution_status(self):
        return 'Success' if not self.error_messages else 'Error'

    def today(self , format : str = '%Y%m%d'):
        return self.init_time.strftime(format)

    def send_email(self):
        if self.email: 
            title = f'{self.status} - {self.task_name.replace("_", " ").title()} - {self.time_str}'
            bodies = [
                f'Machine : {MACHINE.name}' ,
                f'Source : {self.source}' ,
                f'Task name : {self.task_full_name}' ,
                f'Start time : {self.init_time.strftime("%Y-%m-%d %H:%M:%S")}' ,
                f'End time : {self.end_time.strftime("%Y-%m-%d %H:%M:%S")}' ,
                f'AutoRun Status : {self.status}' ,
                f'Excution Status : {self.execution_status}' ,
                f'Error Messages : ' + '-' * 20 ,
                self.error_message ,
                f'Final Messages : ' + '-' * 20 ,
                self.final_message ,
            ]

            self.emailer.send(title , '\n'.join(bodies) , confirmation_message='Autorun' , 
                              attachment_group = ['default' , 'autorun'])

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

    def attach(self , file : Path | str | list[Path] | list[str] , streamlit = True , email = True):
        if not isinstance(file , list): file = [Path(file)]
        file = [Path(f) for f in file]
        if streamlit: self.exit_files.extend([f for f in file if f not in self.exit_files])
        if email: self.emailer.Attach(file , group = 'autorun')
        
    @property
    def final_message(self):
        return '\n'.join(self.logged_messages)
    
    @property
    def error_message(self):
        return '\n'.join(self.error_messages)
    
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