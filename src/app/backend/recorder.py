import os  , traceback

from typing import Any , Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from src.app.abc import argparse_dict
from .task import TaskItem , TaskDatabase

class BackendTaskRecorder:
    '''
    convert script main function to one that can be used as a task in streamlit project
    use BackendTaskRecorder to record task, take acceptable return as exit message
    params will be passed to the warpped function as kwargs, example:
        @BackendTaskRecorder(x = 1)
        def test(x : int , **kwargs):
            return 'yes' , Path('test.txt') , Path('test.csv')
            return BackendTaskRecorder.ExitMessage(message = 'yes' , files = ['test.txt' , 'test.csv'])
            return {'message' : 'yes' , 'files' : ['test.txt' , 'test.csv']}

    track:
        if True, will track the task and update the task status in the _page_files anyway
        if False, will only track the task if task_id is passed through the kwargs or argparse
    kwargs params:
        will passed to the task as kwargs , except for task_id and running_status (will be used by the recorder)
        task_id:
            will be used to identify the task
        running_status:
            pid , status , end_time , exit_code , exit_error , exit_message , exit_files
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
        parsed_kwargs = self.parse_kwargs(kwargs)
        task_id : str | None = parsed_kwargs.pop('task_id' , None)
        if task_id:
            self._task_id = task_id
        else:
            task_item = TaskItem.create(None , source=parsed_kwargs.get('source' , None) , queue = True)
            self._task_id = task_item.id
        self.update_msg : dict[str , Any] = {'pid': os.getpid()}
        self.params = parsed_kwargs
        if 'email' in self.params:
            if isinstance(self.params['email'] , str): 
                self.params['email'] = eval(self.params['email'])
            self.params['email'] = bool(self.params['email'])

    def __repr__(self):
        return f'BackendTaskRecorder(task_id = {self.task_id})'
    
    def __call__(self , func : Callable):
        def wrapper(*args , **kwargs):
            with self:
                ret = func(*args , **kwargs , **self.params)
                self._func_return(ret)
            return ret
        return wrapper

    def __getitem__(self , key : str):
        if key == 'task_id':
            return self.task_id
        elif key in self.update_msg:
            return self.update_msg[key]
        else:
            return self.params.get(key , None)
        
    def __setitem__(self , key : str , value : Any):
        if key == 'task_id':
            raise ValueError('task_id is read only')
        elif key in self.update_msg:
            self.update_msg[key] = value
        else:
            self.params[key] = value

    @staticmethod
    def parse_kwargs(kwargs : dict[str , Any]) -> dict[str , Any]:
        kwargs = argparse_dict(**kwargs)
        for key, value in kwargs.items():
            if not isinstance(value , str):
                kwargs[key] = value
            elif value.lower() in ['true' , 'false']:
                kwargs[key] = eval(value.lower().capitalize())
            elif value.lower() in ['null' , 'none']:
                kwargs[key] = None
            elif value.isdigit():
                kwargs[key] = eval(value)
            else:
                kwargs[key] = value
        return kwargs

    @property
    def task_id(self):
        '''task_id : script_name@time_id'''
        return self._task_id

    def __enter__(self):
        return self

    def __exit__(self , exc_type , exc_value , exc_traceback):
        self.update_msg['end_time'] = datetime.now().timestamp()
        if exc_type is None:
            self.update_msg['status'] = 'error' if self.exit_msg.code else 'complete'
        else:
            self.update_msg['status'] = 'error'
            self.update_msg['exit_code'] = 1
            self.update_msg['exit_message'] = str(exc_value)
            self.update_msg['exit_error'] = traceback.format_exc()
        TaskDatabase().update_task(self.task_id, backend_updated = True, **self.update_msg)

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
        def from_return(cls , ret : Any | None = None):
            if ret is None:
                return cls()
            elif isinstance(ret , cls):
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
                return cls(message = ret.exit_message , files = ret.exit_files , 
                           code = len(ret.error_messages) , error = '\n'.join(ret.error_messages))
            else:
                return cls(message = str(ret))

    def _func_return(self , func_return : Any | None = None):
        if not self.task_id: 
            return
        exit_msg = self.ExitMessage.from_return(func_return)
        self.exit_msg = exit_msg
        if exit_msg.message: 
            self.update_msg['exit_message'] = exit_msg.message
        if exit_msg.files:   
            self.update_msg['exit_files'] = [str(f) for f in exit_msg.files]
        if exit_msg.code:    
            self.update_msg['exit_code'] = exit_msg.code
        if exit_msg.error:   
            self.update_msg['exit_error'] = exit_msg.error

