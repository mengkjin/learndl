from typing import Any, Callable
import inspect
from src.basic import AutoRunTask
from src.app.script_lock import ScriptLockMultiple
from src.app.backend import BackendTaskRecorder

def _get_default_args(func):
    """get default args of a function"""
    sig = inspect.signature(func)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
class ScriptTool:
    """
    Tool to wrap script to be used as a task in streamlit project
    example:
        @ScriptTool('test_streamlit')
        def test_streamlit(port_name : str , **kwargs):
            print(port_name)

        # equivalent to:
        @BackendTaskRecorder(txt = 'Bye, World!' , email = 0)
        @ScriptLockMultiple('test_streamlit' , lock_num = 2 , timeout = 2)
        @AutoRunTask('test_streamlit' , '@port_name')
        def test_streamlit(port_name : str , **kwargs):
            print(port_name)
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
    def info(cls , message : str):
        return AutoRunTask.info(message)

    @classmethod
    def error(cls , message : str):
        return AutoRunTask.error(message)

    @classmethod
    def warning(cls , message : str):
        return AutoRunTask.warning(message)

    @classmethod
    def debug(cls , message : str):
        return AutoRunTask.debug(message)

    @classmethod
    def critical(cls , message : str):
        return AutoRunTask.critical(message)

    @classmethod
    def get_value(cls , key : str):
        return AutoRunTask.get_value(key)
