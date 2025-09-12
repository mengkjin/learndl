import sys
from typing import Any, Literal , IO
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path

class OutputDeflector:
    """double output stream: output to console and catcher"""
    def __init__(
            self, 
            type : Literal['stdout' , 'stderr'] ,
            catcher : 'OutputDeflector | IO | OutputCatcher | None', 
            keep_original : bool = True,
            catcher_function : str = 'write',
        ):
        self.catcher = catcher
        self.type = type

        self.keep_original = keep_original
        self.catcher_function = catcher_function
        self._null_initialize()

    def __repr__(self):
        return f'{self.__class__.__name__}(original={self.original_output}, catcher={self.catcher}, type={self.type})'
    
    def _null_initialize(self):
        self.original_output = None
        self.is_catching = False
        if hasattr(self, '_catcher_write'):
            del self._catcher_write
        if hasattr(self, '_catcher_flush'):
            del self._catcher_flush

    @property
    def original_std(self):
        if self.original_output is not None:
            return self.original_output
        elif self.type == 'stdout':
            return sys.stdout
        elif self.type == 'stderr':
            return sys.stderr
        else:
            raise ValueError(f"Invalid type: {self.type}")
        
    def catcher_write(self , text : str | Any):
        if not hasattr(self, '_catcher_write'):
            self._catcher_write = getattr(self.catcher, self.catcher_function , lambda *x: None)
        self._catcher_write(text)

    def catcher_flush(self):
        if not hasattr(self, '_catcher_flush'):
            self._catcher_flush = getattr(self.catcher, 'flush' , lambda: None)
        self._catcher_flush()

    def original_write(self, text : str | Any):
        self.original_std.write(text)

    def original_flush(self):
        self.original_std.flush()

    def start_catching(self):
        if self.type == 'stdout':
            self.original_output = sys.stdout
            sys.stdout = self
        elif self.type == 'stderr':
            self.original_output = sys.stderr
            sys.stderr = self
        else:
            raise ValueError(f"Invalid type: {self.type}")
        self.is_catching = True
        return self
    
    def end_catching(self):
        if self.type == 'stdout':
            sys.stdout = self.original_output
        elif self.type == 'stderr':
            sys.stderr = self.original_output
        else:
            raise ValueError(f"Invalid type: {self.type}")
        self.close()
        self._null_initialize()
        return self
            
    def write(self, text : str | Any):
        # output to console
        if not self.is_catching or self.keep_original:
            self.original_write(text)
        if self.is_catching:
            self.catcher_write(text)
        self.flush()
        
    def flush(self):
        self.original_std.flush()
        if self.is_catching:
            self.catcher_flush()

    def close(self):
        if isinstance(self.catcher, IO):
            try:
                self.catcher.close()
            except Exception as e:
                print(f"Error closing catcher: {e}")
                raise e

class OutputCatcher(ABC):
    keep_original : bool = True

    def __init__(self):
        self.stdout_catcher = None
        self.stderr_catcher = None
    
    def __enter__(self):
        self.stdout_deflector = OutputDeflector('stdout', self.stdout_catcher , self.keep_original).start_catching()
        self.stderr_deflector = OutputDeflector('stderr', self.stderr_catcher , self.keep_original).start_catching()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stdout_deflector.end_catching()
        self.stderr_deflector.end_catching()

    def write(self, text : str | Any):
        ...
    
    def flush(self):
        ...

    @abstractmethod
    def get_contents(self) -> Any:
        ...

    @property
    def contents(self) -> Any:
        return self.get_contents()
    
class IOCatcher(OutputCatcher):
    keep_original = False
    
    def __init__(self):
        self.stdout_catcher = StringIO()
        self.stderr_catcher = StringIO()
    
    def get_contents(self):
        return {
            'stdout': self.stdout_catcher.getvalue(),
            'stderr': self.stderr_catcher.getvalue(),
        }
    
    def clear(self):
        self.stdout_catcher.seek(0)
        self.stdout_catcher.truncate(0)
        
        self.stderr_catcher.seek(0)
        self.stderr_catcher.truncate(0)

class LogWriter(OutputCatcher):
    '''change print target to both terminal and file'''
    def __init__(self, log_path : str | Path | None = None):
        self.log_path = log_path
        if log_path is None: 
            self.log_file = None
        else:
            log_path = Path(log_path)
            log_path.parent.mkdir(exist_ok=True,parents=True)
            self.log_file = open(log_path, "w")
        self.stdout_catcher = self.log_file
        self.stderr_catcher = self.log_file

    def get_contents(self):
        if self.log_path is None: 
            return ''
        else:
            with open(self.log_path , 'r') as f:
                return f.read()