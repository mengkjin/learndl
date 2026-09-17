"""Stdout/stderr capture to memory, logs, HTML, markdown, and warning interception."""
from __future__ import annotations
from datetime import datetime
from functools import cached_property
from typing import Literal , TypeAlias
from uuid import uuid4

from src.proj.env import PATH , MACHINE
from src.proj.core import Elapsed , Since
from .basic import OutputCatcher , DeflectorGroup , MarkdownWriter
from .crash_protector import CrashProtectorCatcher

__all__ = ['MarkdownCatcher']

ContentSeperatingBy : TypeAlias = Literal['min', 'hour', 'day'] | None
 
class MarkdownCatcher(OutputCatcher):
    """
    Archive the local crash log and rename the optional live shared markdown at exit.
    A supplied crash_protector must be active and outlive this catcher; otherwise
    standalone use creates its own local crash protector.
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
        crash_protector: CrashProtectorCatcher | None = None,
        **kwargs
    ):
        self.category = category
        self.init_time = init_time if init_time else datetime.now()
        self.title = title
        self.add_time_to_title = add_time_to_title
        
        self.add_export_file(self.export_dir.joinpath(self.category.replace(' ' , '_') , self.filename))
        self.share_filename = None
        if to_share_folder and PATH.share_folder is not None:
            self.share_filename = PATH.share_folder.joinpath('markdown_catcher' , self.filename)
            self.add_export_file(self.share_filename)
        
        self.kwargs = kwargs
        self.seperating_by: ContentSeperatingBy = seperating_by
        self.crash_protector = crash_protector
        self._owns_crash_protector = False
        self.markdown_file = None
        self.markdown_writer = None
        self.running_filename = None

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
        
        # Standalone use (or a disabled task crash catcher) still needs a local source.
        if self.crash_protector is None or self.crash_protector.task_id is None:
            self.crash_protector = CrashProtectorCatcher(
                f'markdown_{uuid4().hex}', init_time=self.init_time,
                seperating_by=self.seperating_by,
            )
            self._owns_crash_protector = True
        if self._owns_crash_protector:
            self.crash_protector.__enter__()
        self.open_markdown_file()
        self.deflectors = DeflectorGroup(self , self.keep_original).start_catching()
        self.logger.remark(f"{self.keyword_repr()}, Capturing Start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.crash_protector is not None
        try:
            try:
                self.close_markdown_file()
            finally:
                self.deflectors.end_catching()
            self.export()
        except BaseException:
            self.crash_protector.retain_file = True
            raise
        finally:
            if self._owns_crash_protector:
                self.crash_protector.__exit__(exc_type, exc_val, exc_tb)
    
    def open_markdown_file(self):
        """Open a unique live shared log; it is never the local archive source."""
        if self.share_filename is None:
            return
        i = 0
        try:
            self.share_filename.parent.mkdir(exist_ok=True, parents=True)
            while True:
                suffix = '.running.md' if i == 0 else f'.{i}.running.md'
                self.running_filename = self.share_filename.with_suffix(suffix)
                try:
                    self.markdown_file = self.running_filename.open('x', encoding='utf-8')
                    break
                except FileExistsError:
                    i += 1
            self.markdown_writer = MarkdownWriter(self.markdown_file, self.seperating_by)
            self.markdown_writer.header(self.full_title)
        except OSError as e:
            self.close()
            self.markdown_writer = None
            self.running_filename = None
            self.logger.error(f"Failed to open shared markdown {self.share_filename}: {e}")

    def close_markdown_file(self):
        """generate the markdown footer , including the finish time, duration, and stats of the catcher, then flush and close the file"""
        assert self.crash_protector is not None
        finish_time = datetime.now()
        kwargs = {
            'finish at' : finish_time,
            'duration' : Elapsed(finish_time - self.start_time).fmtstr,
            'stdout lines' : self.stats['stdout_lines'],
            'stderr lines' : self.stats['stderr_lines'],
        }
        self.crash_protector.markdown_writer.footer(**kwargs)
        if self.markdown_writer is not None:
            try:
                self.markdown_writer.footer(**kwargs)
            except OSError as e:
                self.markdown_writer = None
                self.logger.error(f"Failed to finish shared markdown {self.running_filename}: {e}")
        self.close()
        
    def export(self):
        """Copy the local crash log to archives; only rename the shared live log."""
        assert self.crash_protector is not None
        self.logger.remark(f"{self.keyword_repr()}, Capturing Finished, Cost {Since(self.start_time)}")
        for filename in self.export_file_list:
            if filename == self.share_filename:
                if self.running_filename is None:
                    continue
                try:
                    self.running_filename.replace(filename)
                except OSError as e:
                    self.logger.error(f"Failed to rename {self.running_filename} to {filename}: {e}")
                    continue
            elif not self.crash_protector.export_to(filename):
                continue
            self.logger.footnote(f"result saved to {filename}")
    
    def write_stdout(self, text : str):
        """Write stdout to the markdown file"""
        self.write_shared(text)
        self.stats['stdout_lines'] += 1

    def write_stderr(self, text : str):
        """Write stderr to the markdown file"""
        self.write_shared(text, stderr=True)
        self.stats['stderr_lines'] += 1

    def write_shared(self, text: str, stderr: bool = False):
        """A shared-folder failure must not interrupt local logging or the task."""
        if self.markdown_writer is not None:
            try:
                self.markdown_writer.write(text, stderr=stderr)
            except OSError as e:
                self.markdown_writer = None
                self.logger.error(f"Failed to write shared markdown {self.running_filename}: {e}")

    def get_contents(self):
        if self.crash_protector is not None and self.crash_protector.filename.exists():
            return self.crash_protector.filename.read_text(encoding='utf-8')
        for filename in self.export_file_list:
            if filename.exists():
                return filename.read_text(encoding='utf-8')
        return ''
        
    def close(self):
        self.markdown_writer = None
        if self.markdown_file is not None and not self.markdown_file.closed:
            try:
                self.markdown_file.close()
            except OSError as e:
                self.logger.error(f"Failed to close shared markdown {self.running_filename}: {e}")
