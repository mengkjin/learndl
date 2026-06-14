"""Stdout/stderr capture to memory, logs, HTML, markdown, and warning interception."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from src.proj.env import PATH
from src.proj.core import Since

from .basic import OutputCatcher , DeflectorGroup , MarkdownWriter

__all__ = ['CrashProtectorCatcher']

class CrashProtectorCatcher(OutputCatcher):
    """
    Crash protector catcher for stdout and stderr, export to crash protector file in runtime, and then remove the crash protector file at exit
    ideally should be put before all other catchers (exit after all other catchers)
    example:
        catcher = CrashProtectorCatcher()
        with catcher:
            Logger.stdout('This will be caught')
        contents = catcher.contents
    """
    export_dir = PATH.runtime.joinpath('crash_protector')
    export_suffix : str = '.md'

    def __init__(self, task_id : str | None = None, init_time: datetime | None = None, 
                 seperating_by: Literal['min' , 'hour'  , 'day'] | None = 'min', **kwargs):
        self.task_id = task_id
        self.init_time = init_time if init_time else datetime.now()
        
        self.filename = self.export_dir / f'{self.init_time.strftime("%Y%m%d")}.{str(self.task_id).replace('/' , '_')}.md'
    
        self.kwargs = kwargs
        self.seperating_by = seperating_by

    def keyword_repr(self):
        return f'task_id="{self.task_id}"'

    def __enter__(self):
        if self.task_id is None:
            return self
        self.start_time = datetime.now()
        self.open_markdown_file()
        self.deflectors = DeflectorGroup(self , self.keep_original).start_catching()
        self.logger.remark(f"{self.keyword_repr()}, Capturing Start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.task_id is None:
            return
        self.deflectors.end_catching()
        self.logger.remark(f"{self.keyword_repr()}, Capturing Finished, Cost {Since(self.start_time)}")
        self.logger.footnote(f"crash protector file {self.filename} removed")
        self.is_catching = False
    
    def open_markdown_file(self):
        """
        Open the running markdown file , Generate the markdown header , 
        including the title, start time, and basic information of the catcher
        """
        self.filename.parent.mkdir(exist_ok=True,parents=True)
        self.markdown_file = open(self.filename, 'w', encoding='utf-8')
        self.markdown_writer = MarkdownWriter(self.markdown_file, self.seperating_by)
        self.markdown_writer.header(f'{self.task_id} initiated at {self.init_time}')
    
    def write_stdout(self, text):
        """Write stdout to the crash protector file"""
        self.markdown_writer.write(text)
            
    def write_stderr(self, text):
        """Write stderr to the crash protector file"""
        self.markdown_writer.write(text , stderr = True)

    def get_contents(self):
        return ''
        
    def close(self):
        self.markdown_file.close()
        self.filename.unlink(missing_ok=True)
