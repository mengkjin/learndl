"""Stdout/stderr capture to memory, logs, HTML, markdown, and warning interception."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from src.proj.env import PATH , Proj
from src.proj.core import strPath

from .basic import OutputCatcher

__all__ = ['LogWriter']

class LogWriter(OutputCatcher):
    """
    Output catcher into log file for stdout and stderr
    example:
        catcher = LogWriter('log.txt')
        with catcher:
            Logger.stdout('This will be caught')
        contents = catcher.contents
    """
    def __init__(self, log_path : strPath | None = None):
        self.log_path = Path(log_path) if log_path is not None else None
        if log_path is None: 
            self.log_writer = None
        else:
            log_path = PATH.main.joinpath(log_path)
            log_path.parent.mkdir(exist_ok=True,parents=True)
            self.log_writer = open(log_path, "w")
        self.catchers = {'stdout': self.log_writer, 'stderr': self.log_writer}

    def __enter__(self):
        super().__enter__()
        Proj.log_writer = self.log_writer
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        Proj.log_writer = None
        super().__exit__(exc_type, exc_val, exc_tb)

    def write_stdout(self, text : str | Any):
        """Write to the output catcher"""
        if self.log_writer is None:
            return
        self.log_writer.write(text)
        self.log_writer.flush()

    def write_stderr(self, text : str | Any):
        """Write to the output catcher"""
        if self.log_writer is None:
            return
        self.log_writer.write(text)
        self.log_writer.flush()

    def get_contents(self):
        if self.log_path is None: 
            return ''
        else:
            return self.log_path.read_text(encoding='utf-8')