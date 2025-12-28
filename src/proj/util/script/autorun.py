import traceback , sys

from datetime import datetime
from pathlib import Path
from typing import Any , Callable

from src.proj.env import MACHINE , Proj
from src.proj.log import Logger
from src.proj.calendar import CALENDAR
from src.proj.func import print_disk_space_info

from .task_record import TaskRecorder
from ..email import Email
from ..catcher import HtmlCatcher , MarkdownCatcher , WarningCatcher


class TaskName:
    def __init__(self):
        self._name = None

    def __get__(self , instance, owner = None):
        if instance is None and owner is not None:
            return owner._instances[-1].task_name
        assert self._name is not None , 'TaskName is not set'
        return self._name.replace(' ' , '_').lower()

    def __set__(self , instance, value : str):
        self._name = value

class TaskKey:
    def __init__(self):
        self._key = None

    def __bool__(self):
        return self._key is not None

    def __get__(self , instance : 'AutoRunTask', owner = None):
        if self._key is None:
            return None
        elif isinstance(self._key , str) and self._key.startswith('@'):
            key = self._key.removeprefix('@')
            if instance is not None:
                value = instance[key]
            else:
                value = owner.get_value(key)
            return str(value).lower()
        else:
            return str(self._key).lower()

    def __set__(self , instance, value):
        self._key = value

class AutoRunCatchers:
    _catch_warnings = [
        'must accept context and return_scalar arguments' ,
        'an item of incompatible dtype' ,
    ]
    def __init__(self , catchers : list[str] = ['html' , 'markdown' , 'warning']):
        self.catchers = catchers

    def enter(self , title : str , category : str , init_time : datetime):
        self._catchers = []
        if 'html' in self.catchers:
            self._catchers.append(HtmlCatcher(title , category , init_time))
        if 'markdown' in self.catchers:
            self._catchers.append(MarkdownCatcher(title , category , init_time , 
                                  to_share_folder=True , add_time_to_title=False))
        if 'warning' in self.catchers:
            self._catchers.append(WarningCatcher(self._catch_warnings))

        for catcher in self._catchers:
            catcher.__enter__()
        return self

    def exit(self, *args):
        for catcher in self._catchers[::-1]:
            catcher.__exit__(*args)
        return self

class AutoRunTask:
    """
    AutoRunTask manager for common tasks
    features:
        1. catch messages to html/markdown
        2. catch warnings of given list and yield exception
        3. record task status to database for streamlit app
        4. send email with attachment if in server and email is True
    example:
        with AutoRunTask('daily_update' , 'CALENDAR.update_to()'):
            Logger.stdout('This is the task...')
        with AutoRunTask('daily_update' , '@source'):
            Logger.stdout('This is the task...')
    """
    task_name = TaskName()
    task_key = TaskKey()
    _instances = []

    def __new__(cls , *args , **kwargs):
        if cls not in cls._instances:
            cls._instances.append(super().__new__(cls))
        return cls._instances[-1]

    def __init__(
        self , 
        task_name : str , 
        task_key : str | Any | None = None , 
        catchers : list[str] = ['html' , 'markdown' , 'warning'],
        forfeit_if_done = False,
        verbosity : int | None = None ,
        **kwargs
    ):
        self.task_name = task_name
        self.task_key = task_key
        self.init_time = datetime.now()
        
        self.catchers = AutoRunCatchers(catchers)
        self.forfeit_if_done = forfeit_if_done

        self.verbosity = verbosity

        self.kwargs = kwargs
        
        self.exit_files = []
        self.error_messages = []
        
        self.status = 'Starting'
        self.func_return : Any | None = None

    def __bool__(self):
        return True

    def __repr__(self):
        return f'AutoRunTask(task_name={self.task_name},task_key={self.task_key},email={self.email},source={self.source},time={self.time_str})'

    def __enter__(self):
        self.task_recorder = TaskRecorder('autorun' , self.task_name , self.task_key or '')
        self.already_done = self.task_recorder.is_finished()
        self.catchers.enter(self.task_full_name , self.task_name , self.init_time)
        self.status = 'Running'
        self.set_verbosity()
        Proj.print_info(script_level = True)
        Logger.divider()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = datetime.now()

        if exc_type is not None:
            traceback.print_exc()
            self.status = 'Error'
            self.error_messages.append('\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        else:
            self.status = 'Success'

        self.error_messages.extend(Logger.get_conclusions('error'))
        self.exit_message = Logger.draw_conclusions()

        print_disk_space_info()
        
        self.catchers.exit(exc_type, exc_value, exc_traceback)
        
        self.exit_files = list(set(self.exit_files + [Path(f) for f in Proj.States.exit_files]))
        Proj.States.exit_files.clear()

        # send email if not forfeit task
        if not self.forfeit_task:
            self.send_email()

        if self.execution_success: 
            self.task_recorder.mark_finished(
                remark = ' | '.join([f'source: {self.source}' , 
                                     f'exit_code: {len(self.error_messages)}']))

    def __call__(self , func : Callable):
        assert callable(func) , 'func must be a callable'
        def wrapper(*args , **kwargs):
            self.kwargs.update(kwargs)
            with self:
                if self.forfeit_if_done and self.forfeit_task:
                    Logger.conclude(f'task {self.task_full_name} is forfeit, most likely due to finished autoupdate, skip daily update' , level = 'error')
                else:
                    self.func_return = func(*args , **self.kwargs)
            return self
        return wrapper

    def get(self , key : str) -> Any:
        """get the value of the key from the object or kwargs"""
        if key in dir(self):
            raw = getattr(self , key)
        else:
            raw = self.kwargs[key]
        try:
            if isinstance(raw , str): 
                raw = eval(raw)
        except Exception:
            pass
        return raw
    
    def __getitem__(self , key : str):
        if key in dir(self):
            raw = getattr(self , key)
        else:
            raw = self.kwargs[key]
        try:
            if isinstance(raw , str): 
                raw = eval(raw)
        except Exception:
            pass
        return raw
        
    @property
    def task_full_name(self) -> str:
        return self.task_name if not self.task_key else f'{self.task_name}_{self.task_key}'

    @property
    def task_title(self) -> str:
        title = self.task_name.replace('_', ' ').title()
        if self.task_key:
            title = f'{title} of {self.task_key}'
        return title

    @property
    def time_str(self) -> str:
        return self.init_time.strftime('%Y%m%d%H%M%S')

    @property
    def email(self) -> bool:
        return bool(self.kwargs.get('email' , True))

    @property
    def source(self) -> str:
        return self.kwargs.get('source' , 'py')
    
    @property
    def manual_start(self) -> bool:
        """return True if the task script is not done"""
        return not self.source == 'bash'

    @property
    def forfeit_task(self) -> bool:
        """return True if the task script is already done and the source is bash , so crontab scripts can be forfeit"""
        return self.already_done and not self.manual_start

    @property
    def success(self) -> bool:
        """return True if the task script is successfully run"""
        return self.status == 'Success'

    @property
    def execution_status(self) -> str:
        """return True if the task script is failed"""
        return 'Error' if self.error_messages else self.status

    @property
    def execution_success(self) -> bool:
        """
        return the execution status of the task script
        will be 'Success' only if no error messages , otherwise 'Error'
        more strict than status
        """
        return not self.error_messages
   
    @property
    def error_message(self) -> str:
        """return the aggregated error message of the task"""
        return '\n'.join(self.error_messages)
    
    def today(self , format : str = '%Y%m%d') -> str:
        """return the today's date in the given format"""
        return self.init_time.strftime(format)

    def set_verbosity(self):
        """set the verbosity of the task"""
        if 'max_vb' in self.kwargs:
            self.verbosity = 10 * int(self.kwargs['max_vb'])
        Proj.verbosity = self.verbosity

    def send_email(self):
        """send email with attachment if in server and email is True"""
        if self.email: 
            title = f'{self.execution_status} - {self.task_name.replace("_", " ").title()} - {self.time_str}'
            bodies = [
                f'Machine : {MACHINE.name}' ,
                f'Source : {self.source}' ,
                f'Task name : {self.task_full_name}' ,
                f'Start time : {self.init_time.strftime("%Y-%m-%d %H:%M:%S")}' ,
                f'End time : {self.end_time.strftime("%Y-%m-%d %H:%M:%S")}' ,
                f'AutoRun Status : {self.status}' ,
                f'Excution Status : {self.execution_status}' ,
                f'CMD line : {" ".join(sys.argv)}' ,
                f'Error Messages : ' + '-' * 20 ,
                self.error_message ,
                f'Final Messages : ' + '-' * 20 ,
                self.exit_message ,
            ]
            Email.send(title , '\n'.join(bodies) , confirmation_message='Autorun' , additional_attachments = self.exit_files)

    @classmethod
    def get_value(cls , key : str) -> Any:
        """get the value of the key from the object or kwargs"""
        obj = cls._instances[-1]
        return obj[key]

    @classmethod
    def update_to(cls) -> int:
        """return the desired update to date"""
        return CALENDAR.update_to()
        
    