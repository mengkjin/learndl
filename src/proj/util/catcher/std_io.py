"""Stdout/stderr capture to memory, logs, HTML, markdown, and warning interception."""
from __future__ import annotations

from io import StringIO
from typing import Any
from .basic import OutputCatcher

__all__ = ['IOCatcher']

class IOCatcher(OutputCatcher):
    """
    Output catcher into StringIO for stdout and stderr
    example:
        catcher = IOCatcher()
        with catcher:
            Logger.stdout('This will be caught')
        contents = catcher.contents
    """
    keep_original = False
    
    def __init__(self):
        self.stdout_catcher : StringIO = StringIO()
        self.stderr_catcher : StringIO = StringIO()

    def write_stdout(self, text : str | Any):
        """Write to the output catcher"""
        self.stdout_catcher.write(text)

    def write_stderr(self, text : str | Any):
        """Write to the output catcher"""
        self.stderr_catcher.write(text)
        
    def get_contents(self):
        self.stdout_catcher.flush()
        self.stderr_catcher.flush()
        return {
            'stdout': self.stdout_catcher.getvalue(),
            'stderr': self.stderr_catcher.getvalue()
        }    
    
    def clear(self):
        """Clear the contents of the output catcher"""
        self.stdout_catcher.seek(0)
        self.stderr_catcher.seek(0)
        self.stdout_catcher.truncate(0)
        self.stderr_catcher.truncate(0)