"""Decorator to expose callables as named Streamlit tasks with locking and default kwargs."""

import inspect , os
from pathlib import Path

from functools import wraps , cached_property
from typing import Any , Callable , Literal

from src.proj.env import PATH
from src.proj.log import Logger
from src.proj.util.parallel import is_main_process


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
        forfeit_if_done : bool = False ,
        lock_name : str | None = None ,
        lock_num : int = 1 , 
        lock_timeout : int = 60 , 
        markdown_catcher : bool = False ,
        verbosity : int | None = None ,
        source_mode : Literal['script' , 'api'] = 'script' ,
        interaction : dict[str, Any] | None = None ,
        **kwargs
    ):  
        self.task_name = task_name
        self.task_key = task_key
        self.lock_name = lock_name
        self.source_mode = source_mode
        self.interaction = interaction
        self._recorder_kwargs = kwargs
        self._forfeit_if_done = forfeit_if_done
        self._markdown_catcher = markdown_catcher
        self._verbosity = verbosity

        ln , lt = lock_num , lock_timeout
        if source_mode == 'api' and interaction:
            if 'lock_num' in interaction:
                ln = int(interaction['lock_num'])
            if interaction.get('lock_timeout') is not None:
                lt = int(interaction['lock_timeout'])
        self._lock_num = ln
        self._lock_timeout = lt

        os.chdir(str(PATH.main))

    @cached_property
    def backend_recorder(self):
        """Get the backend recorder"""
        from src.interactive.backend import BackendTaskRecorder
        backend_recorder = BackendTaskRecorder(**self._recorder_kwargs).resolve_task_id()
        return backend_recorder

    @cached_property
    def script_lock(self):
        """Get the script lock"""
        from src.proj.util.script.script_lock import ScriptLockMultiple
        return ScriptLockMultiple(self.lock_name or self.task_name , self._lock_num , self._lock_timeout)

    @cached_property
    def autorun_task(self):
        """Get the autorun task"""
        from src.proj.util.script.autorun import AutoRunTask
        return AutoRunTask(
            self.task_name , self.task_key , self._forfeit_if_done , self._verbosity ,
            task_id = self.backend_recorder.task_id , markdown_catcher = self._markdown_catcher,
        )

    def __call__(self , func : Callable):
        assert callable(func), 'func must be a callable'

        from src.api.contract import filter_kwargs_explicit_only
        sig = inspect.signature(func)
        @wraps(func)
        def inner(*args: Any , **kwargs: Any) -> Any:
            """Strip ``email`` / ``task_id`` / etc. so ``AutoRunTask`` metadata never reaches API callables."""
            return func(*args , **filter_kwargs_explicit_only(sig , kwargs))

        @wraps(func)
        def wrapper(*args , **kwargs):
            if not is_main_process():
                return inner(*args , **kwargs)
            self.autorun_task.kwargs = _get_default_args(func) | self.autorun_task.kwargs
            if self.source_mode == 'api':
                data = self.interaction
                if data is None:
                    from src.api.contract import endpoint_schema
                    data = endpoint_schema(func)
                if data is not None and 'email' in data:
                    self.autorun_task.kwargs['email'] = bool(data['email'])
                elif 'email' not in self.autorun_task.kwargs:
                    self.autorun_task.kwargs['email'] = False
            else:
                if 'email' not in self.autorun_task.kwargs and (caller_path := _get_caller_path()) is not None:
                    from src.interactive.backend import ScriptHeader
                    self.autorun_task.kwargs.update({'email' : ScriptHeader.read_from_file(caller_path).email})
            new_func = self.backend_recorder(self.script_lock(self.autorun_task(inner)))
            return new_func(*args , **kwargs)
        return wrapper

    def __repr__(self):
        return f'ScriptTool(task_name={self.task_name},task_key={self.task_key},lock_name={self.lock_name})'

    @property
    def task_id(self):
        return self.backend_recorder.task_id
