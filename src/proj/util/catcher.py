"""Stdout/stderr capture to memory, logs, HTML, markdown, and warning interception."""

import sys , re , platform , warnings , shutil
import pandas as pd

from typing import Any ,Literal , IO , Union , Callable
from abc import ABC, abstractmethod
from io import StringIO , TextIOWrapper
from pathlib import Path
from dataclasses import dataclass , field
from datetime import datetime
from matplotlib.figure import Figure
from string import Template

from src.proj.env import PATH , MACHINE
from src.proj.env import Proj
from src.proj.core import Duration , str_to_html , dataframe_to_html , figure_to_html
from src.proj.log import Logger

__all__ = [
    'IOCatcher' , 'LogWriter' , 'OutputCatcher' , 'OutputDeflector' , 
    'HtmlCatcher' , 'MarkdownCatcher' , 'CrashProtectorCatcher' , 'WarningCatcher' ,
]

type_of_std = Literal['stdout' , 'stderr']
type_of_catcher = Union['OutputDeflector' , IO , 'OutputCatcher' , None]

class OutputDeflector:
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
            type : type_of_std ,
            catcher : type_of_catcher, 
            keep_original : bool = True,
        ):
        self.type = type
        self.catcher = catcher
        self.keep_original = keep_original
        self.original = None
        self.is_catching = False

    def __repr__(self):
        return f'{self.__class__.__name__}(original={self.original}, catcher={self.catcher}, type={self.type})'

    def get_write_flush(self , output : type_of_catcher) -> tuple[Callable, Callable]:
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
                Logger.error(f"Error closing catcher: {e}")
                raise

class DeflectorGroup:
    """Pair of ``OutputDeflector``s for stdout and stderr."""

    def __init__(self , catcher : 'OutputCatcher' , 
                 keep_original : bool = True):
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
class OutputCatcher(ABC):
    """
    Abstract base class for output catcher for stdout and stderr
    must implement the get_contents method
    can rewrite the stdout_catcher/stderr_catcher or stdout_deflector/stderr_deflector
    """
    keep_original : bool = True
    export_suffix : str = '.log'

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

    @property
    def contents(self) -> Any:
        """Get the contents of the output catcher"""
        if not hasattr(self , '_contents'):
            self._contents = self.get_contents()
        return self._contents

    @contents.setter
    def contents(self, value : Any) -> None:
        """Override cached contents (e.g. after context exit)."""
        self._contents = value

    @property
    def export_file_list(self) -> list[Path]:
        """Get the export file list, usually the default path and the user provided path"""
        if not hasattr(self , '_export_file_list'):
            self._export_file_list : list[Path] = []
        return self._export_file_list

    def set_export_files(self , export_file_list : list[Path | str] | Path | str | None = None):
        """Replace export targets with normalized ``Path`` list."""
        self.export_file_list.clear()
        if export_file_list is None:
            ...
        elif isinstance(export_file_list , list):
            self.export_file_list.extend([Path(path) for path in export_file_list])
        else:
            self.export_file_list.append(Path(export_file_list))

    def add_export_file(self , export_path : Path | str | None = None):
        """Add an path to the export file list"""
        if export_path is None:
            return
        export_path = Path(export_path) if isinstance(export_path ,  str) else export_path
        assert export_path.suffix == self.export_suffix , f"export_path must be a {self.export_suffix} file , but got {export_path}"
        self.export_file_list.append(export_path)
    
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

class LogWriter(OutputCatcher):
    """
    Output catcher into log file for stdout and stderr
    example:
        catcher = LogWriter('log.txt')
        with catcher:
            Logger.stdout('This will be caught')
        contents = catcher.contents
    """
    def __init__(self, log_path : str | Path | None = None):
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

class WarningCatcher:
    """
    catch specific warnings and show call stack
    example:
        with WarningCatcher(['This will raise an exception']):
            raise Exception('This will raise an exception')
    """
    def __init__(self , catch_warnings : list[str] | None = None , * ,
                 method : Literal['raise' , 'ignore'] = 'raise' ,
                 highlight_varibles : dict[str, Any] | None = None):
        self.method = method
        self.warnings_caught = []
        self.original_showwarning = warnings.showwarning
        warnings.filterwarnings('always')
        self.catch_warnings = [] if catch_warnings is None else [c.lower() for c in catch_warnings]
        self.highlight_varibles = highlight_varibles
    
    def custom_showwarning(self, message, category, filename, lineno, file=None, line=None) -> None:
        """Custom warning show function to catch specific warnings and show call stack"""
        # only catch the warnings we care about
        if any(c in str(message).lower() for c in self.catch_warnings):
            Logger.alert1(f"\n caught warning: {message}")
            Logger.alert1(f"warning location: {filename}:{lineno}")
            Logger.alert1("call stack:")
            Logger.print_traceback_stack(color = 'lightyellow' , bold = True)
            Logger.alert1("-" * 80)

            if self.highlight_varibles is not None:
                for var_name, var_value in self.highlight_varibles.items():
                    Logger.alert1(f"{var_name}: {var_value}")
                Logger.alert1("-" * 80)
                
            if self.method == 'raise':
                raise Exception(message)
        
        # call original warning show function
        self.original_showwarning(message, category, filename, lineno, file, line)
    
    def __enter__(self):
        warnings.showwarning = self.custom_showwarning
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.showwarning = self.original_showwarning

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
        content_str = str(self.content) if not isinstance(self.content , Figure) else 'matplotlib figure'
        return f'{self.time_str}|{self.type_str}|vb{self.vb_level_str}|{content_str}'
    
    @classmethod
    def create(cls, content: str | pd.DataFrame | pd.Series | Figure | None | Any , output_type: str | None = None):
        """Create a timed output item"""
        infos = {}
        valid = True
        if output_type is None:
            if isinstance(content , Figure): 
                output_type = 'figure'
            elif isinstance(content , pd.DataFrame): 
                output_type = 'data_frame'
            else: 
                if not isinstance(content , str):
                    Logger.warning(f"Unknown output type for catcher.TimedOutput.create: {type(content)}")
                    content = str(content)
                output_type = 'stdout'
        if output_type == 'stderr':
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
        elif output_type == 'stdout':
            assert isinstance(content, str) , f"content must be a string , but got {type(content)}"
            if content == '...': 
                valid = False

        return cls(output_type, content , infos , valid)
    
    def equivalent(self, other: 'TimedOutput') -> bool:
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
        template = Template(HtmlCatcher.Templates['row'].safe_substitute(
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
        
class HtmlCatcher(OutputCatcher):
    """
    Html catcher for stdout, stderr, dataframe, and image (use Logger.display to display), export to html file at exit
    example:
        catcher = HtmlCatcher()
        with catcher:
            Logger.stdout('This will be caught')
        contents = catcher.contents
    """
    export_dir = PATH.logs.joinpath('catcher' , 'html')
    export_suffix : str = '.html'

    PrimaryInstance : 'HtmlCatcher | None' = None
    Capturing : bool = True
    Templates : dict[str, Template] = PATH.load_templates('html' , 'html_catcher')

    def __init__(self, title: str | None = None , category : str = 'miscelaneous', init_time: datetime | None = None , 
                 add_time_to_title: bool = True, **kwargs):
        self.category = category
        self.init_time = init_time if init_time else datetime.now()
        self.title = title
        self.add_time_to_title = add_time_to_title

        self.add_export_file(self.export_dir.joinpath(self.category.replace(' ' , '_') , self.filename))
        
        self.outputs: list[TimedOutput] = []
        self.kwargs = kwargs
        
    def __bool__(self):
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}(title={self.title},primary={self.is_primary})"

    @property
    def title(self) -> str:
        """Get the export file list of the catcher"""
        if not hasattr(self , '_title'):
            self._title = None
        title = self._title if self._title else 'html_catcher'
        if self.add_time_to_title:
            time_str = self.init_time.strftime("%Y%m%d%H%M%S")
            title = f'{title} at {time_str}'
        return title

    @title.setter
    def title(self , value : str | None):
        """Set the title of the catcher"""
        self._title = value

    @property
    def filename(self) -> str:
        """Get the filename of the catcher"""
        return f'{self.title.replace(" " , "_")}.html'

    @property
    def is_running(self):
        """Check if the catcher is running"""
        return self.PrimaryInstance is not None

    @property
    def is_primary(self):
        """Check if the catcher is the primary instance"""
        return self.__class__.PrimaryInstance is self

    @property
    def start_point(self) -> int | None:
        """Get the start point of output index in the primary instance's output list"""
        if not hasattr(self , '_start_point'):
            self._start_point = None
        return self._start_point

    @start_point.setter
    def start_point(self , value : int):
        """Set the start point of output index in the primary instance's output list"""
        assert not self.is_primary , f"Start point can only be set for secondary instances"
        self._start_point = value

    def set_attrs(self , title : str | None = None , category : str | None = None):
        """
        Set the attributes of the catcher even after initialization
        title : str , the title of the catcher
        category : str , the category of the catcher
        """
        instance = self.PrimaryInstance if self.PrimaryInstance is not None else self
        if title: 
            instance.title = title
        if category: 
            instance.category = category
        return self

    def set_instance(self):
        """Set the instance of the catcher, if the catcher is already running, block the new instance"""
        if self.__class__.PrimaryInstance is None:
            self.__class__.PrimaryInstance = self

    def clear_instance(self):
        """Clear the instance of the catcher if the catcher is the current instance"""
        if self.__class__.PrimaryInstance is self:
            self.__class__.PrimaryInstance = None

    def __enter__(self):
        self.set_instance()
        self.start_time = datetime.now()
        if self.is_primary:
            self.deflectors = DeflectorGroup(self , self.keep_original).start_catching()
            self.redirect_display_function()
        else:
            assert self.PrimaryInstance is not None , f"Primary instance is not set when entering {self}"
            self.start_point = len(self.PrimaryInstance.outputs)

        Logger.remark(f"{self} Capturing Start" , vb_level = 1 if self.is_primary else 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Logger.remark(f"{self} Capturing Finished, cost {Duration(since = self.start_time)}" , vb_level = 1 if self.is_primary else 2)
        self.export()
        if self.is_primary:
            self.deflectors.end_catching()
            self.restore_display_function()
        self.clear_instance()

    def export(self):
        """Export the catcher to all paths in the export file list"""
        # log first and then export
        for export_path in self.export_file_list:
            Logger.footnote(f"{self.__class__.__name__} result saved to {export_path}" , indent = 1 , vb_level = 3)
        if self.is_primary and self.export_file_list:
            Proj.exit_files.insert(0 , self.export_file_list[0])
        
        html_content = self.generate_html()
        for export_path in self.export_file_list:
            export_path.parent.mkdir(exist_ok=True,parents=True)
            export_path.write_text(html_content, encoding='utf-8')
        
    def redirect_display_function(self):
        """redirect Logger.Display to catcher"""
        Logger.set_display_callbacks([self.add_output , self.stop_capturing] , [self.start_capturing])

    def restore_display_function(self):
        """restore Logger.Display functions"""
        Logger.reset_display_callbacks()
    
    def generate_html(self):
        """generate html file with time ordered outputs"""
        assert self.PrimaryInstance is not None , f"Primary instance is not set when generating html"

        templates = [output.to_template() for output in self.PrimaryInstance.outputs[self.start_point:]]
        templates = [template for template in templates if template is not None]
        rows = [templates.substitute(index=i) for i, templates in enumerate(templates)]

        return ''.join([self.html_head() , *rows , self.html_tail()])
 
    def add_output(self, content: str | pd.DataFrame | pd.Series | Figure | Any , output_type: str | None = None):
        """add output to time ordered list"""
        if not self.Capturing: 
            return
        
        output = TimedOutput.create(content , output_type)
        if not output or (self.outputs and output.equivalent(self.outputs[-1])): 
            return

        self.outputs.append(output)

    @classmethod
    def stop_capturing(cls , *args, **kwargs):
        """Stop the capturing of the catcher , class level (stop all catchers)"""
        cls.Capturing = False

    @classmethod
    def start_capturing(cls , *args, **kwargs):
        """Start the capturing of the catcher , class level (start all catchers)"""
        cls.Capturing = True

    def write_stdout(self, text: str):
        """Write stdout to the catcher"""
        if text := text.strip('\n'):
            self.add_output(text, 'stdout')

    def write_stderr(self, text: str):
        """Write stderr to the catcher"""
        if text := text.strip('\n'):
            self.add_output(text, 'stderr')

    def get_contents(self):
        """Get the contents of the html catcher"""
        return self.generate_html()
       
    def html_head(self) -> str:
        """Generate the html head , including the styles of the html file, and basic information of the catcher"""
        title = self.title.title()
        
        key_width = 80
        if self.kwargs:
            key_width = max(int(max(len(key) for key in list(self.kwargs.keys())) * 5.5) + 10 , key_width)
        finish_time = datetime.now()

        script_infos = {
            'Machine' : MACHINE.name,
            'Python' : f"{platform.python_version()}-{platform.machine()}",
            'Command' : ' '.join(sys.argv),
            'Start at' : f'{self.start_time.strftime("%Y-%m-%d %H:%M:%S")}',
            'Finish at' : f'{finish_time.strftime("%Y-%m-%d %H:%M:%S")}',
            'Duration' : Duration((finish_time - self.start_time).total_seconds()).fmtstr,
        }
        other_types : list[str] = list(set([output.type_str for output in self.outputs if output.type not in ['stdout' , 'stderr' , 'dataframe' , 'image']]))
        output_infos = {
            'Total #' : len(self.outputs),
            'Stdout #' : sum(1 for output in self.outputs if output.type == 'stdout'),
            'Stderr #' : sum(1 for output in self.outputs if output.type == 'stderr'),
            'Dataframe #' : sum(1 for output in self.outputs if output.type == 'dataframe'),
            'Image #' : sum(1 for output in self.outputs if output.type == 'image'),
            **{type.title() + ' #' : sum(1 for output in self.outputs if output.type_str == type) for type in other_types},
        }
        output_infos = {key: value for key, value in output_infos.items() if value > 0}
        infos_script = '<div class="add-infos add-title"> INFORMATION </div>' + \
            '\n'.join([f'<div class="add-infos"><span class="add-key">{key}</span><span class="add-seperator">:</span><span class="add-value">{value}</span></div>' 
                       for key, value in script_infos.items()])
        infos_outputs = '<div class="add-infos add-title"> NUMBER OF OUTPUTS </div>' + \
            '\n'.join([f'<div class="add-infos"><span class="add-key">{key}</span><span class="add-seperator">:</span><span class="add-value">{value}</span></div>' 
                        for key, value in output_infos.items()])
        head = self.Templates['head'].substitute(
            title=title,
            key_width=key_width,
            infos_script=infos_script,
            infos_outputs=infos_outputs,
        )
        return head

    @classmethod
    def html_tail(cls) -> str:
        """Generate the html tail , including the end of the html file"""
        return cls.Templates['tail'].substitute() 

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

    def __init__(self, title: str | None = None,
                 category : str = 'miscelaneous',
                 init_time: datetime | None = None,
                 add_time_to_title: bool = False,
                 to_share_folder: bool = MACHINE.cuda_server ,
                 seperating_by: Literal['min' , 'hour'  , 'day'] | None = 'min',
                 **kwargs):

        self.category = category
        self.init_time = init_time if init_time else datetime.now()
        self.title = title
        self.add_time_to_title = add_time_to_title
        
        self.add_export_file(self.export_dir.joinpath(self.category.replace(' ' , '_') , self.filename))
        if to_share_folder and PATH.share_folder is not None:
            self.add_export_file(PATH.share_folder.joinpath('markdown_catcher' , self.filename))
        
        self.kwargs = kwargs
        self.seperating_by = seperating_by
        

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(title={self.title})'

    @property
    def title(self) -> str:
        """Get the export file list of the catcher"""
        if not hasattr(self , '_title'):
            self._title = None
        title = self._title if self._title else 'markdown_catcher'
        if self.add_time_to_title:
            time_str = self.init_time.strftime("%Y%m%d%H%M%S")
            title = f'{title} at {time_str}'
        return title

    @title.setter
    def title(self , value : str | None):
        """Set the title of the catcher"""
        self._title = value

    @property
    def filename(self) -> str:
        """Get the filename of the catcher"""
        return f'{self.title.replace(" " , "_")}.md'

    def __enter__(self):
        self.start_time = datetime.now()
        
        self.stats = {
            'stdout_lines' : 0,
            'stderr_lines' : 0,
        }
        
        self.open_markdown_file()
        self.deflectors = DeflectorGroup(self , self.keep_original).start_catching()
        Logger.remark(f"{self} Capturing Start" , vb_level = 1)
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
                Logger.error(f"Too many running markdown files, max_running_files={max_running_files}")
                running_filename = self.export_file_list[-1].with_suffix('.running.md')
                break
        self.running_filename = running_filename
        self.running_filename.parent.mkdir(exist_ok=True,parents=True)
        self.markdown_file = open(self.running_filename, 'w', encoding='utf-8')
        self.markdown_writer = MarkdownWriter(self.markdown_file, self.seperating_by)

        self.markdown_writer.header(self.title)

    def close_markdown_file(self):
        """generate the markdown footer , including the finish time, duration, and stats of the catcher, then flush and close the file"""
        finish_time = datetime.now()
        kwargs = {
            'finish at' : finish_time,
            'duration' : Duration((finish_time - self.start_time).total_seconds()).fmtstr,
            'stdout lines' : self.stats['stdout_lines'],
            'stderr lines' : self.stats['stderr_lines'],
        }
        self.markdown_writer.footer(**kwargs)
        self.markdown_file.close()
        
    def export(self):
        """Export the running markdown file to the export file list, and then delete the running file"""
        Logger.remark(f"{self} Capturing Finished, cost {Duration(since = self.start_time)}" , vb_level = 1)
        self.markdown_file.close()
        for filename in self.export_file_list:
            filename.unlink(missing_ok=True)
            filename.parent.mkdir(exist_ok=True,parents=True)
            try:
                shutil.copy(self.running_filename, filename)
            except OSError as e:
                Logger.error(f"Failed to copy {self.running_filename} to {filename}: {e}")
            Logger.footnote(f"{self.__class__.__name__} result saved to {filename}" , indent = 1 , vb_level = 3)
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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(task_id={self.task_id})'

    def __enter__(self):
        if self.task_id is None:
            return self
        self.start_time = datetime.now()
        self.open_markdown_file()
        self.deflectors = DeflectorGroup(self , self.keep_original).start_catching()
        Logger.remark(f"{self} Capturing Start" , vb_level = 1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.task_id is None:
            return
        self.deflectors.end_catching()
        Logger.remark(f"{self} Capturing Finished, cost {Duration(since = self.start_time)}" , vb_level = 1)
        Logger.footnote(f"{self.__class__.__name__} file {self.filename} removed" , indent = 1 , vb_level = 3)
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
