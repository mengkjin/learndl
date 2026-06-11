"""Stdout/stderr capture to memory, logs, HTML, markdown, and warning interception."""
from __future__ import annotations
import sys , re , platform
import pandas as pd

from abc import ABC, abstractmethod
from dataclasses import dataclass , field
from datetime import datetime
from functools import cached_property
from io import TextIOWrapper
from pathlib import Path
from string import Template
from typing import Any , Literal , Callable , TextIO,  TYPE_CHECKING

from src.proj.env import PATH , Proj , MACHINE
from src.proj.core import str_to_html , dataframe_to_html , figure_to_html , strPath
from src.proj.log import Logger
from src.proj.bases import BoundLogger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    'OutputCatcher' , 'OutputDeflector' , '_get_html_templates' , 'TimedOutput'
]

_HtmlTemplates : dict[str, Template] = {} 
def _get_html_templates(key : str):
    if not _HtmlTemplates:
        _HtmlTemplates.update(PATH.load_templates('html' , 'html_catcher'))
    return _HtmlTemplates[key]

class OutputDeflector(BoundLogger):
    """
    double output stream: deflect output to catcher and original output stream (optional)
    example:
        catcher = IOCatcher()
        with OutputDeflector('stdout', catcher, keep_original=True):
            Logger.stdout('This will be deflected to catcher')
        with OutputDeflector('stderr', catcher, keep_original=False):
            Logger.info('This will be deflected to catcher')
    """
    def __init__(
        self, 
        type : Literal['stdout' , 'stderr'] ,
        catcher : OutputDeflector | OutputCatcher | None, 
        keep_original : bool = True,
        * , indent: int = 0 , vb_level: int = 1 , **kwargs
    ):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.type = type
        self.catcher = catcher
        self.keep_original = keep_original
        self.original = None
        self.is_catching = False

    def keyword_repr(self):
        return f'original={self.original}, catcher={self.catcher}, type={self.type}'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.keyword_repr()})'

    def get_write_flush(self , output : OutputDeflector | OutputCatcher | TextIO | None) -> tuple[Callable, Callable]:
        if output is None:
            return lambda *x: None, lambda: None
        elif isinstance(output , OutputCatcher):
            if self.type == 'stdout':
                return output.write_stdout, output.flush_stdout
            else:
                return output.write_stderr, output.flush_stderr
        else:
            return output.write, output.flush

    def start_catching(self):
        """
        Start catching of the output stream
        1. redirect stdout/stderr to the deflector
        2. set the deflector to catching mode
        """
        if self.type == 'stdout':
            self.original = sys.stdout
            sys.stdout = self
        elif self.type == 'stderr':
            self.original = sys.stderr
            sys.stderr = self
        else:
            raise ValueError(f"Invalid type: {self.type}")
        self.is_catching = True
        self.catcher_write, self.catcher_flush = self.get_write_flush(self.catcher)
        self.original_write, self.original_flush = self.get_write_flush(self.original)
        return self
    
    def end_catching(self):
        """
        End catching of the output stream
        1. reset stdout/stderr to original output stream
        2. close the deflector
        3. null initialize the deflector
        """
        if self.type == 'stdout':
            sys.stdout = self.original
        elif self.type == 'stderr':
            sys.stderr = self.original
        else:
            raise ValueError(f"Invalid type: {self.type}")
        self.close()
        return self
            
    def write(self, text : str | Any) -> None:
        """Write to the catcher (if catching) and original output stream (if keep_original)"""
        if not self.is_catching or self.keep_original:
            self.original_write(text)
        if self.is_catching:
            self.catcher_write(text)
            self.catcher_flush()
        
    def flush(self) -> None:
        """Flush the original output stream and the catcher"""
        self.original_flush()
            
    def close(self) -> None:
        """Close the catcher"""
        if hasattr(self.catcher, 'close'):
            try:
                getattr(self.catcher, 'close')()
            except Exception as e:
                self.logger.error(f"Error closing catcher: {e}")
                raise

class DeflectorGroup:
    """Pair of ``OutputDeflector``s for stdout and stderr."""

    def __init__(self , catcher : OutputCatcher , keep_original : bool = True):
        self.stdout = OutputDeflector('stdout', catcher, keep_original)
        self.stderr = OutputDeflector('stderr', catcher, keep_original)

    def start_catching(self):
        """Begin redirection for both streams."""
        self.stdout.start_catching()
        self.stderr.start_catching()
        return self

    def end_catching(self):
        """Restore both streams."""
        self.stdout.end_catching()
        self.stderr.end_catching()
class OutputCatcher(BoundLogger , ABC):
    """
    Abstract base class for output catcher for stdout and stderr
    must implement the get_contents method
    can rewrite the stdout_catcher/stderr_catcher or stdout_deflector/stderr_deflector
    """
    keep_original : bool = True
    export_suffix : str = '.log'

    def keyword_repr(self):
        return f'keep_original={self.keep_original}'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.keyword_repr()})'

    def __enter__(self):
        """Enter the output catcher , start stdout and stderr redirection"""
        self.deflectors = DeflectorGroup(self , self.keep_original).start_catching()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the output catcher , end stdout and stderr redirection"""
        self.contents = self.get_contents()
        self.deflectors.end_catching()

    @abstractmethod
    def write_stdout(self, text : str | Any):
        """Write to the output catcher, if write_stdout/write_stderr is defined, write to the corresponding catcher"""
        
    @abstractmethod
    def write_stderr(self, text : str | Any):
        """Write to the output catcher, if write_stdout/write_stderr is defined, write to the corresponding catcher"""
    
    def flush_stdout(self):
        """Flush the stdout of the output catcher"""
        ...
    
    def flush_stderr(self):
        """Flush the output catcher"""
        ...

    @abstractmethod
    def get_contents(self) -> Any:
        """Get the contents of the output catcher"""
        ...

    @cached_property
    def contents(self) -> Any:
        """Get the contents of the output catcher"""
        return self.get_contents()

    @cached_property
    def export_file_list(self) -> list[Path]:
        """Get the export file list, usually the default path and the user provided path"""
        return []

    def set_export_files(self , export_file_list : list[strPath] | strPath | None = None):
        """Replace export targets with normalized ``Path`` list."""
        self.export_file_list.clear()
        if export_file_list is None:
            ...
        elif isinstance(export_file_list , list):
            self.export_file_list.extend([Path(path) for path in export_file_list])
        else:
            self.export_file_list.append(Path(export_file_list))

    def add_export_file(self , export_path : strPath | None = None):
        """Add an path to the export file list"""
        if export_path is None:
            return
        export_path = Path(export_path) if isinstance(export_path ,  str) else export_path
        assert export_path.suffix == self.export_suffix , f"export_path must be a {self.export_suffix} file , but got {export_path}"
        self.export_file_list.append(export_path)
 
@dataclass
class TimedOutput:
    """time ordered output item"""
    type: Literal['stdout' , 'stderr' , 'data_frame' , 'figure'] | str
    content: str | pd.DataFrame | pd.Series | Figure | None | Any
    infos: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    
    def __post_init__(self):
        self._time = datetime.now()
        self.type_fmt = self.get_type_fmt()
        self.type_str = self.get_type_str()
        self.vb_level = Proj.vb.vb_level

    def __bool__(self):
        return self.valid
    
    def get_type_fmt(self) -> Literal['stdout' , 'stderr' , 'dataframe' , 'image'] | str:
        match self.type:
            case 'stdout':
                return 'stdout'
            case 'stderr':
                return 'stderr'
            case 'data_frame':
                return 'dataframe'
            case 'figure':
                return 'image'
            case _:
                return 'other'
    
    def get_type_str(self) -> Literal['STDERR' , 'STDOUT' , 'TABLE' , 'IMAGE'] | str:
        match self.type:
            case 'stderr':
                return 'STDERR'
            case 'stdout':
                return 'STDOUT'
            case 'data_frame':
                return 'TABLE'
            case 'figure':
                return 'IMAGE'
            case _:
                return self.type.upper()

    @property
    def create_time(self):
        """Get the creation time of the output item"""
        return self._time.timestamp()

    @property
    def sort_key(self):
        """Get the sort key of the output item : timestamp"""
        return self._time.timestamp()

    @property
    def time_str(self) -> str:
        """Get the time string of the output item"""
        return self._time.strftime('%H:%M:%S.%f')[:-3]

    @property
    def vb_level_str(self) -> str:
        """Get the vb level string of the output item"""
        return f'' if self.vb_level is None or self.vb_level == 0 else f'{self.vb_level}'

    @property
    def crash_protector_str(self) -> str:
        """Get the crash protector string of the output item"""
        content_str = str(self.content) if not self.content.__class__.__name__ == 'Figure' else 'matplotlib figure'
        return f'{self.time_str}|{self.type_str}|vb{self.vb_level_str}|{content_str}'
    
    @classmethod
    def create(cls, content: str | pd.DataFrame | pd.Series | Figure | None | Any , output_type: str | None = None):
        """Create a timed output item"""
        infos = {}
        valid = True
        assert output_type not in ['stdout' , 'stderr'] or isinstance(content, str) , f"{output_type} content must be a string , but got {type(content)}"
        if isinstance(content, str) and (content.strip() == '' or content == '...'):
            content = ''
            valid = False
            output_type = output_type or 'stdout'
        elif output_type is None:
            if isinstance(content , pd.DataFrame): 
                output_type = 'data_frame'
            elif content.__class__.__name__ == 'Figure':
                output_type = 'figure'
            else: 
                if not isinstance(content , (str , bool , int , float , Path)):
                    Logger.warning(f"Unknown output type for catcher.TimedOutput.create: {type(content)}")
                content = str(content)
                output_type = 'stdout'
        elif output_type == 'stderr':
            assert isinstance(content, str) , f"content must be a string , but got {type(content)}"
            r0 = re.search(r"^(.*?)(\d{1,3})%\|", content) # capture text before XX%|
            r1 = re.search(r"(\d+)/(\d+)", content)  # XX/XX
            r2 = re.search(r"\s*([^\]]+),\s*([^\]]+)it/s]", content) # [XXXXXXX, XXXXit/s]
            if r0 and r1 and r2: # is a progress bar
                infos['is_progress_bar'] = True
                infos['unique_content'] = str(r0.group(1)).strip()
                if infos['unique_content'] == '':
                    valid = False
                elif int(r0.group(2)) != 100 or (int(r1.group(1)) != int(r1.group(2))): # not finished
                    valid = False
                content = content.strip()
                output_type = 'tqdm'

        return cls(output_type, content , infos , valid)
    
    def equivalent(self, other: TimedOutput) -> bool:
        """
        Check if the output item is equivalent to the other item
        equivalent means the content is very similar , or the progress bar is the same
        """
        if self.type in ['data_frame' , 'figure']:
            return False
        return self.type == other.type and str(self.content) == str(other.content)
    
    def to_template(self):
        """Convert the output item to html"""
        if self.content is None: 
            text = None
        elif self.type == 'data_frame':
            text = dataframe_to_html(self.content)
        elif self.type == 'figure':
            text = figure_to_html(self.content)
        else:
            text = str_to_html(self.content)
        if text is None: 
            return None
        
        template = Template(_get_html_templates('row').safe_substitute(
            type_fmt=self.type_fmt,
            type_str=self.type_str,
            vb_level_fmt=self.type_fmt,
            vb_level_str=self.vb_level_str,
            time_str=self.time_str,
            text=text,
        ))
        return template

    def to_html2(self , index : int) -> str | None:
        """Convert the output item to html"""
        template = self.to_template()
        if template is None:
            return None
        return template.substitute(index=index)

    def to_record(self) -> dict[str, Any]:
        """JSON-serializable record for multiprocessing worker log export."""
        if self.type in ('data_frame' , 'figure'):
            content = f'<{self.type} omitted in mp export>'
        else:
            content = str(self.content) if self.content is not None else ''
        return {
            'type': self.type ,
            'content': content ,
            'valid': self.valid ,
            'infos': self.infos ,
            'timestamp': self._time.timestamp() ,
            'vb_level': self.vb_level ,
        }

    @classmethod
    def from_record(cls , record: dict[str, Any] , prefix: str = '') -> TimedOutput:
        """Rebuild from :meth:`to_record` (preserves original timestamp for sorting)."""
        obj = cls(
            record['type'] ,
            prefix + record['content'] ,
            record.get('infos') or {} ,
            record.get('valid' , True) ,
        )
        obj._time = datetime.fromtimestamp(record['timestamp'])
        obj.vb_level = record.get('vb_level')
        obj.type_fmt = obj.get_type_fmt()
        obj.type_str = obj.get_type_str()
        return obj

class MarkdownWriter:
    """Append timestamped HTML-safe lines to a markdown stream with optional section headers."""

    def __init__(self, md_file: TextIOWrapper, seperating_by: str| None = 'min'):
        self.md_file = md_file
        self.seperating_by = seperating_by
        if seperating_by is None:
            self._seperator_time_str = lambda x: ''
            self._prefix_time_str = lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        elif seperating_by == 'min':
            self._seperator_time_str = lambda x: x.strftime('%Y-%m-%d %H:%M')
            self._prefix_time_str = lambda x: x.strftime('%H:%M:%S.%f')[:-3]
        elif seperating_by == 'hour':
            self._seperator_time_str = lambda x: x.strftime('%Y-%m-%d %H')
            self._prefix_time_str = lambda x: x.strftime('%H:%M:%S.%f')[:-3]
        elif seperating_by == 'day':
            self._seperator_time_str = lambda x: x.strftime('%Y-%m-%d')
            self._prefix_time_str = lambda x: x.strftime('%H:%M:%S.%f')[:-3]
        else:
            raise ValueError(f"Invalid SeperatingBy: {seperating_by}")
        self.last_seperator = None

    def write(self, text: str , stderr : bool = False , prefix='- ' , suffix='  \n'):
        """Write text to the markdown file"""
        if not text.strip():
            return
        self.write_seperator()
        text = f'{self._prefix_time_str(datetime.now())}: {text}'
        text = str_to_html(text.strip('\n'))
        if stderr:
            text = f'<u>{text}</u>'
        self.md_file.write(prefix + text + suffix)
        self.md_file.flush()

    def write_seperator(self):
        """Write the seperator of the markdown file"""
        if self.seperating_by is None: 
            return
        seperator = self._seperator_time_str(datetime.now())
        if seperator != self.last_seperator:
            self.md_file.write(f"### {seperator} \n")
            self.last_seperator = seperator

    def header(self , title: str = ''):
        """Write the header of the markdown file"""
        self.md_file.write(f"# {title.title()}\n")
        self.md_file.write(f"## Log Start \n")
        self.md_file.write(f"- *Machine: {MACHINE.name}*  \n")
        self.md_file.write(f"- *Python: {platform.python_version()}-{platform.machine()}*  \n")
        self.md_file.write(f"- *Start at: {datetime.now()}*  \n")
        self.md_file.write(f"## Log Main \n")
        self.md_file.flush()

    def footer(self , **kwargs):
        """Write the footer of the markdown file"""
        self.md_file.write(f"## Log End \n")
        for key , value in kwargs.items():
            self.md_file.write(f"- *{key.title()}: {value}*  \n")
        self.md_file.write(f"***\n")
        self.md_file.flush()
