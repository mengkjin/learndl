import traceback

from datetime import datetime
from pathlib import Path
from typing import Any

from src.proj import MACHINE , PATH , Logger , HtmlCatcher , MarkdownCatcher , WarningCatcher
from .calendar import CALENDAR
from .task_record import TaskRecorder
from .email import Email

_catch_warnings = ['must accept context and return_scalar arguments']

class AutoRunTask:
    def __init__(self , task_name : str , task_key : str | Any | None = None , email = True , message_catcher : bool = True , source = 'py' , **kwargs):
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

        self.html_catcher = HtmlCatcher.CreateCatcher(self.message_catcher_path if message_catcher else False , 
                                                         self.task_full_name , self.init_time)
        self.md_catcher = MarkdownCatcher(self.task_full_name , to_share_folder=True , add_time_to_title=False)
        self.warning_catcher = WarningCatcher(_catch_warnings)

        self.update_to = CALENDAR.update_to()
        self.task_recorder = TaskRecorder('autorun' , self.task_name , self.task_key or '')
        self.emailer = Email()
        self.logger = Logger()

        self.status = 'Starting'

    def __repr__(self):
        return f'AutoRunTask(task_name = {self.task_full_name} , email = {self.email} , source = {self.source} , time = {self.time_str})'

    def __enter__(self):
        self.already_done = self.task_recorder.is_finished()
        # change_power_mode('balanced')
        self.html_catcher.__enter__()
        self.md_catcher.__enter__()
        self.warning_catcher.__enter__()
        self.status = 'Running'
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = datetime.now()
        for level , msgs in self.exit_messages.items():
            for msg in msgs:
                getattr(self.logger , level)(msg)
        if exc_type is not None:
            traceback.print_exc()
            self.status = 'Error'
            self.error_messages.append('\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        else:
            self.status = 'Success'
        
        self.warning_catcher.__exit__(exc_type, exc_value, exc_traceback)
        self.md_catcher.__exit__(exc_type, exc_value, exc_traceback)
        self.html_catcher.__exit__(exc_type, exc_value, exc_traceback)
        
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
            if isinstance(raw , str): 
                raw = eval(raw)
        except Exception:
            pass
        return raw
    
    def __getitem__(self , key : str):
        return self.kwargs[key]
        
    @property
    def forfeit_task(self):
        return self.already_done and self.source == 'bash'
    
    @property
    def message_catcher_path(self):
        return PATH.log_autorun.joinpath('message_catcher' , f'{self.task_full_name}.{self.time_str}.html')
    
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
        if not isinstance(file , list): 
            file = [Path(file)]
        file = [Path(f) for f in file]
        if streamlit: 
            self.exit_files.extend([f for f in file if f not in self.exit_files])
        if email: 
            self.emailer.Attach(file , group = 'autorun')
        
    @property
    def final_message(self):
        return '\n'.join(self.logged_messages)
    
    @property
    def error_message(self):
        return '\n'.join(self.error_messages)