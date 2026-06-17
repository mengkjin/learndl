"""Stdout/stderr capture to memory, logs, HTML, markdown, and warning interception."""
from __future__ import annotations
import shutil

from datetime import datetime
from functools import cached_property
from typing import Literal , TypeAlias

from src.proj.env import PATH , MACHINE
from src.proj.core import Elapsed , Since
from .basic import OutputCatcher , DeflectorGroup , MarkdownWriter

__all__ = ['MarkdownCatcher']

ContentSeperatingBy : TypeAlias = Literal['min', 'hour', 'day'] | None
 
class MarkdownCatcher(OutputCatcher):
    """
    Markdown catcher for stdout and stderr, export to running markdown file in runtime, and then export to markdown file at exit
    example:
        catcher = MarkdownCatcher()
        with catcher:
            Logger.stdout('This will be caught')
        contents = catcher.contents
    """
    export_dir = PATH.logs.joinpath('catcher' , 'markdown')
    export_suffix : str = '.md'

    def __init__(
        self, title: str | None = None,
        category : str = 'miscelaneous',
        init_time: datetime | None = None,
        add_time_to_title: bool = False,
        to_share_folder: bool = MACHINE.cuda_server ,
        seperating_by: ContentSeperatingBy = 'min',
        **kwargs
    ):
        self.category = category
        self.init_time = init_time if init_time else datetime.now()
        self.title = title
        self.add_time_to_title = add_time_to_title
        
        self.add_export_file(self.export_dir.joinpath(self.category.replace(' ' , '_') , self.filename))
        if to_share_folder and PATH.share_folder is not None:
            self.add_export_file(PATH.share_folder.joinpath('markdown_catcher' , self.filename))
        
        self.kwargs = kwargs
        self.seperating_by = seperating_by

    def keyword_repr(self):
        return f'title="{self.full_title}"'

    @cached_property
    def title(self) -> str | None:
        """Get the export file list of the catcher"""
        return None

    @cached_property
    def full_title(self) -> str:
        """Get the full title of the catcher"""
        title = self.title or 'markdown_catcher'
        if self.add_time_to_title:
            time_str = self.init_time.strftime("%Y%m%d%H%M%S")
            title = f'{title} at {time_str}'
        return title

    @property
    def filename(self) -> str:
        """Get the filename of the catcher"""
        return f'{self.full_title.replace(" " , "_")}.md'

    def __enter__(self):
        self.start_time = datetime.now()
        
        self.stats = {
            'stdout_lines' : 0,
            'stderr_lines' : 0,
        }
        
        self.open_markdown_file()
        self.deflectors = DeflectorGroup(self , self.keep_original).start_catching()
        self.logger.remark(f"{self.keyword_repr()}, Capturing Start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_markdown_file()
        self.deflectors.end_catching()
        self.export()
    
    def open_markdown_file(self , max_running_files : int = 5):
        """
        Open the running markdown file , Generate the markdown header , 
        including the title, start time, and basic information of the catcher
        """
        i = 0
        running_filename = self.export_file_list[-1].with_suffix('.running.md')
        while running_filename.exists():
            running_filename = self.export_file_list[-1].with_suffix(f'.{i}.running.md')
            i += 1
            if i >= max_running_files:
                self.logger.error(f"Too many running markdown files, max_running_files={max_running_files}")
                running_filename = self.export_file_list[-1].with_suffix('.running.md')
                break
        self.running_filename = running_filename
        self.running_filename.parent.mkdir(exist_ok=True,parents=True)
        self.markdown_file = open(self.running_filename, 'w', encoding='utf-8')
        self.markdown_writer = MarkdownWriter(self.markdown_file, self.seperating_by)

        self.markdown_writer.header(self.full_title)

    def close_markdown_file(self):
        """generate the markdown footer , including the finish time, duration, and stats of the catcher, then flush and close the file"""
        finish_time = datetime.now()
        kwargs = {
            'finish at' : finish_time,
            'duration' : Elapsed(finish_time - self.start_time).fmtstr,
            'stdout lines' : self.stats['stdout_lines'],
            'stderr lines' : self.stats['stderr_lines'],
        }
        self.markdown_writer.footer(**kwargs)
        self.markdown_file.close()
        
    def export(self):
        """Export the running markdown file to the export file list, and then delete the running file"""
        self.logger.remark(f"{self.keyword_repr()}, Capturing Finished, Cost {Since(self.start_time)}")
        self.markdown_file.close()
        for filename in self.export_file_list:
            filename.unlink(missing_ok=True)
            filename.parent.mkdir(exist_ok=True,parents=True)
            try:
                shutil.copy(self.running_filename, filename)
            except OSError as e:
                self.logger.error(f"Failed to copy {self.running_filename} to {filename}: {e}")
            self.logger.footnote(f"result saved to {filename}")
        for path in self.running_filename.parent.glob(f'{self.filename.removesuffix(".md")}.*running.md'):
            path.unlink()
    
    def write_stdout(self, text : str):
        """Write stdout to the markdown file"""
        self.markdown_writer.write(text)
        self.stats['stdout_lines'] += 1

    def write_stderr(self, text : str):
        """Write stderr to the markdown file"""
        self.markdown_writer.write(text , stderr = True)
        self.stats['stderr_lines'] += 1

    def get_contents(self):
        if self.running_filename.exists():
            return self.running_filename.read_text(encoding='utf-8')
        elif self.export_file_list:
            return self.export_file_list[-1].read_text(encoding='utf-8')
        else:
            return ''
        
    def close(self):
        self.markdown_file.close()