import inspect
from pathlib import Path

from typing import Any, Callable

from src.proj import Logger
from src.basic import AutoRunTask
from src.app.script_lock import ScriptLockMultiple
from src.app.backend import BackendTaskRecorder , ScriptHeader

__all__ = ['ScriptTool']

def _get_default_args(func):
    """get default args of a function"""
    sig = inspect.signature(func)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def _get_caller_path(depth : int = 2):
    """get the directory of the caller function"""
    frame = inspect.currentframe()
    path = None
    try:
        assert frame is not None , 'frame is not None'
        caller_frame = frame
        for _ in range(depth):
            caller_frame = getattr(caller_frame , 'f_back')
        assert caller_frame is not None , 'caller_frame is not None'
        caller_info = inspect.getframeinfo(caller_frame)
        path = Path(caller_info.filename).absolute()
    except Exception as e:
        Logger.warning(f'get caller directory failed: {e}')
    finally:
        del frame
    return path

class ScriptTool:
    """
    Tool to wrap script to be used as a task in streamlit project
    example:
        @ScriptTool('test_streamlit')
        def test_streamlit(port_name : str , **kwargs):
            Logger.stdout(port_name)

        # equivalent to:
        @BackendTaskRecorder(txt = 'Bye, World!' , email = 0)
        @ScriptLockMultiple('test_streamlit' , lock_num = 2 , timeout = 2)
        @AutoRunTask('test_streamlit' , '@port_name')
        def test_streamlit(port_name : str , **kwargs):
            Logger.stdout(port_name)
    """
    def __init__(
        self , 
        task_name : str , 
        task_key : str | Any | None = None , 
        catchers : list[str] = ['html' , 'markdown' , 'warning'],
        forfeit_if_done : bool = False ,
        lock_name : str | None = None ,
        lock_num : int = 1 , 
        lock_timeout : int = 60 , 
        lock_wait_time : int = 1 ,
        **kwargs
    ):
        self.task_name = task_name
        self.task_key = task_key
        self.lock_name = lock_name
        
        if 'email' not in kwargs and (caller_path := _get_caller_path()) is not None:
            header = ScriptHeader.read_from_file(caller_path)
            kwargs.update({'email' : header.email})


        self.backend_recorder = BackendTaskRecorder(**kwargs)
        self.script_lock = ScriptLockMultiple(lock_name or task_name , lock_num , lock_timeout , lock_wait_time)
        self.autorun_task = AutoRunTask(task_name , task_key , catchers , forfeit_if_done)

    def __call__(self , func : Callable):
        assert callable(func), 'func must be a callable'
        self.autorun_task.kwargs = _get_default_args(func) | self.autorun_task.kwargs
        return self.backend_recorder(self.script_lock(self.autorun_task(func)))

    def __repr__(self):
        return f'ScriptTool(task_name={self.task_name},task_key={self.task_key},lock_name={self.lock_name})'

    @classmethod
    def get_value(cls , key : str):
        return AutoRunTask.get_value(key)
