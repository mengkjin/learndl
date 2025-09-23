import traceback

from datetime import datetime
from pathlib import Path
from typing import Any , Literal

from src.proj import MACHINE , PATH , Logger , HtmlCatcher , MarkdownCatcher , WarningCatcher
from .calendar import CALENDAR
from .task_record import TaskRecorder
from .email import Email

_catch_warnings = [
    'must accept context and return_scalar arguments' ,
    'an item of incompatible dtype' ,
]

class AutoRunTask:
    """
    AutoRunTask manager for common tasks
    features:
        1. catch messages to html/markdown
        2. catch warnings of given list and yield exception
        3. record task status to database for streamlit app
        4. send email with attachment if in server and email is True
    example:
        with AutoRunTask('task_name'):
            print('This is the task...')
    """
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
        self.logged_messages = []
        self.error_messages = []

        self.html_catcher = HtmlCatcher.CreateCatcher(self.message_catcher_path if message_catcher else False , 
                                                         self.task_full_name , self.init_time)
        self.md_catcher = MarkdownCatcher(self.task_full_name , to_share_folder=True , add_time_to_title=False)
        self.warning_catcher = WarningCatcher(_catch_warnings)

        self.task_recorder = TaskRecorder('autorun' , self.task_name , self.task_key or '')

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
        
        for log_type , message in Logger.iter_cached_messages():
            self.log(log_type, message , cached = False)

        if exc_type is not None:
            traceback.print_exc()
            self.status = 'Error'
            self.error_messages.append('\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        else:
            self.status = 'Success'
        
        self.warning_catcher.__exit__(exc_type, exc_value, exc_traceback)
        self.md_catcher.__exit__(exc_type, exc_value, exc_traceback)
        self.html_catcher.__exit__(exc_type, exc_value, exc_traceback)
        
        self.attach(Email.Attachments.get('default' , []) , streamlit = True , email = False)
        if not self.forfeit_task:
            self.send_email()
            if self.status == 'Success': 
                self.task_recorder.mark_finished(
                    remark = ' | '.join([f'source: {self.source}' , 
                                       f'exit_code: {len(self.error_messages)}']))
        # change_power_mode('power-saver')

    @property
    def update_to(self) -> int:
        """return the desired update to date"""
        return CALENDAR.update_to()

    @property
    def success(self) -> bool:
        """return True if the task script is successfully run"""
        return self.status == 'Success'

    @property
    def execution_status(self) -> Literal['Success' , 'Error']:
        """
        return the execution status of the task script
        will be 'Success' only if no error messages , otherwise 'Error'
        more strict than status
        """
        return 'Success' if not self.error_messages else 'Error'

    def get(self , key : str , default : Any = None) -> Any:
        """get the value of the key from other kwargs"""
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
    def forfeit_task(self) -> bool:
        """return True if the task script is already done and the source is bash , so crontab scripts can be forfeit"""
        return self.already_done and self.source == 'bash'
    
    @property
    def message_catcher_path(self) -> Path:
        """return the path of the html message catcher"""
        return PATH.log_autorun.joinpath('message_catcher' , f'{self.task_full_name}.{self.time_str}.html')
    
    def today(self , format : str = '%Y%m%d') -> str:
        """return the today's date in the given format"""
        return self.init_time.strftime(format)

    def send_email(self):
        """send email with attachment if in server and email is True"""
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
                self.exit_message ,
            ]

            Email.send(title , '\n'.join(bodies) , confirmation_message='Autorun' , 
                       attachment_group = ['default' , 'autorun'])

    def log(self , log_type : Literal['info' , 'warning' , 'error' , 'critical' , 'debug'] , message : str , cached = True):
        """log the message to the logger if cached is False , otherwise cache the message"""
        if cached:
            Logger.cache_message(log_type, message)
        else:
            getattr(Logger , log_type)(message)
            self.logged_messages.append(f'{log_type.upper()} : {message}')

    def info(self , message : str , cached = True):
        self.log('info', message , cached)
    
    def error(self , message : str , cached = True):
        """Additional to log the error message , add the message to the error messages"""
        self.log('error', message , cached)
        self.error_messages.append(message)

    def warning(self , message : str , cached = True):
        self.log('warning', message , cached)

    def debug(self , message : str , cached = True):
        self.log('debug', message , cached)

    def critical(self , message : str , cached = True):
        self.log('critical', message , cached)

    def attach(self , file : Path | str | list[Path] | list[str] , streamlit = True , email = True):
        """attach the file to the task , can select to streamlit app or email"""
        if not isinstance(file , list): 
            file = [Path(file)]
        file = [Path(f) for f in file]
        if streamlit: 
            self.exit_files.extend([f for f in file if f not in self.exit_files])
        if email: 
            Email.Attach(file , group = 'autorun')
        
    @property
    def exit_message(self) -> str:
        """return the aggregated exit message of the task"""
        return '\n'.join(self.logged_messages)
    
    @property
    def error_message(self) -> str:
        """return the aggregated error message of the task"""
        return '\n'.join(self.error_messages)