import io , sys , re , html , base64 , platform , warnings , shutil , traceback
import pandas as pd

from typing import Any ,Literal , IO , Union
from abc import ABC, abstractmethod
from io import StringIO , TextIOWrapper
from pathlib import Path
from dataclasses import dataclass , field
from datetime import datetime
from matplotlib.figure import Figure

from src.proj.env import PATH , MACHINE
from src.proj.proj import Proj
from src.proj.abc import Duration
from src.proj.log import Logger

__all__ = [
    'IOCatcher' , 'LogWriter' , 'OutputCatcher' , 'OutputDeflector' , 
    'HtmlCatcher' , 'MarkdownCatcher' , 'CrashProtectorCatcher' , 'WarningCatcher' ,
]

type_of_std = Literal['stdout' , 'stderr']
type_of_catcher = Union['OutputDeflector' , IO , 'OutputCatcher' , None]

def _str_to_html(text: str | Any):
    """capture string to html"""
    
    assert isinstance(text, str) , f"text must be a string , but got {type(text)}"

    text = html.escape(text)
    text = re.sub(r'(?:\u001b\[[\d;]*m)+', _replace_ansi_sequences, text)
    
    return text

def _replace_ansi_sequences(match):
    """replace ANSI sequences to html span tag"""
    # match.group(0) contains all continuous ANSI sequences
    sequences = match.group(0)
    all_codes = []

    for seq_match in re.finditer(r'\u001b\[([\d;]*)m', sequences):
        codes_str = seq_match.group(1)
        if codes_str:
            all_codes.extend(codes_str.split(';'))
    
    return _convert_ansi_codes_to_span(all_codes)

def _convert_ansi_codes_to_span(codes):
    """convert ANSI codes list to a single span tag"""
    styles = []
    bg_color = None
    fg_color = None

    color_map = {
        # regular foreground colors (30-37)
        30: 'black',
        31: 'red',
        32: 'green',
        33: 'yellow',
        34: 'blue',
        35: 'magenta',  # more standard than 'purple'
        36: 'cyan',
        37: 'white',
        
        # regular background colors (40-47)
        40: 'black',
        41: 'red',
        42: 'green',
        43: 'yellow',
        44: 'blue',
        45: 'magenta',
        46: 'cyan',
        47: 'white',
        
        # bright colors (90-97)
        90: '#7f7f7f',      # bright black / gray
        91: '#ff5555',      # bright red
        92: '#55ff55',      # bright green
        93: '#ffff55',      # bright yellow
        94: '#5555ff',      # bright blue
        95: '#ff55ff',      # bright magenta
        96: '#55ffff',      # bright cyan
        97: '#ffffff',      # bright white
        
        # bright background colors (100-107)
        100: '#7f7f7f',      # bright black
        101: '#ff5555',      # bright red
        102: '#55ff55',      # bright green
        103: '#ffff55',      # bright yellow
        104: '#5555ff',      # bright blue
        105: '#ff55ff',      # bright magenta
        106: '#55ffff',      # bright cyan
        107: '#ffffff',      # bright white
    }
    
    for code_str in codes:
        if not code_str:
            continue
        code = int(code_str)
        if code == 0:
            return '</span>'
        elif code == 1:
            styles.append('font-weight: bold')
        elif code == 3:
            styles.append('font-style: italic')
        elif code in color_map:
            if 30 <= code <= 37 or 90 <= code <= 97:  # foreground colors
                fg_color = color_map[code]
            elif 40 <= code <= 47 or 100 <= code <= 107:  # background colors
                bg_color = color_map[code]
            
    if fg_color:
        styles.append(f'color: {fg_color}')
    if bg_color:
        styles.append(f'background-color: {bg_color}')
        if not fg_color:
            styles.append('color: white')
    
    if styles:
        return f'<span style="{"; ".join(styles)};">'
    return ''

def _figure_to_base64(fig : Figure | Any):
    """convert matplotlib figure to base64 string"""
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64
    except Exception as e:
        Logger.error(f"Error converting figure to base64: {e}")
        return None

def _dataframe_to_html(df: pd.DataFrame | pd.Series | Any):
    """capture display object (dataframe or other object)"""
    assert isinstance(df, (pd.DataFrame , pd.Series)) , f"obj must be a dataframe or series , but got {type(df)}"
    try:
        # get dataframe html representation
        html_table = getattr(df , '_repr_html_')() if hasattr(df, '_repr_html_') else df.to_html(classes='dataframe')
        content = f'<div class="dataframe">{html_table}</div>'
    except Exception:
        # downgrade to text format
        content = f'<div class="df-fallback"><pre>{html.escape(df.to_string())}</pre></div>'
    return content
    
def _figure_to_html(fig: Figure | Any):
    """capture matplotlib figure"""
    assert isinstance(fig, Figure) , f"fig must be a matplotlib figure , but got {type(fig)}"
    content = None
    try:
        if fig.get_axes():  # check if figure has content
            if image_base64 := _figure_to_base64(fig):
                content = f'<img src="data:image/png;base64,{image_base64}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin: 2px 0;">'
    except Exception as e:
        Logger.error(f"Error capturing matplotlib figure: {e}")
        content = f'<div class="figure-fallback"><pre>Error capturing matplotlib figure: {e}</pre></div>'
    return content
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
            catcher_function : str = 'write',
        ):
        self.catcher = catcher
        self.type = type

        self.keep_original = keep_original
        self.catcher_function = catcher_function
        self._null_initialize()

    def __repr__(self):
        return f'{self.__class__.__name__}(original={self.original_output}, catcher={self.catcher}, type={self.type})'
    
    def _null_initialize(self) -> None:
        """Null initialize the deflector"""
        self.original_output = None
        self.is_catching = False
        if hasattr(self, '_catcher_write'):
            delattr(self, '_catcher_write')
        if hasattr(self, '_catcher_flush'):
            delattr(self, '_catcher_flush')

    @property
    def original_std(self) -> 'OutputDeflector | IO | OutputCatcher':
        """Get the original output stream"""
        if self.original_output is not None:
            return self.original_output
        elif self.type == 'stdout':
            return sys.stdout
        elif self.type == 'stderr':
            return sys.stderr
        else:
            raise ValueError(f"Invalid type: {self.type}")
        
    def catcher_write(self , text : str | Any) -> None:
        """Write to the catcher"""
        if not hasattr(self, '_catcher_write'):
            self._catcher_write = getattr(self.catcher, self.catcher_function , lambda *x: None)
        self._catcher_write(text)

    def catcher_flush(self) -> None:
        """Flush the catcher"""
        if not hasattr(self, '_catcher_flush'):
            self._catcher_flush = getattr(self.catcher, 'flush' , lambda: None)
        self._catcher_flush()

    def original_write(self, text : str | Any) -> None:
        """Write to the original output stream"""
        self.original_std.write(text)

    def original_flush(self) -> None:
        """Flush the original output stream"""
        self.original_std.flush()

    def start_catching(self):
        """
        Start catching of the output stream
        1. redirect stdout/stderr to the deflector
        2. set the deflector to catching mode
        """
        if self.type == 'stdout':
            self.original_output = sys.stdout
            sys.stdout = self
        elif self.type == 'stderr':
            self.original_output = sys.stderr
            sys.stderr = self
            Logger.reset_logger()
        else:
            raise ValueError(f"Invalid type: {self.type}")
        self.is_catching = True
        return self
    
    def end_catching(self):
        """
        End catching of the output stream
        1. reset stdout/stderr to original output stream
        2. close the deflector
        3. null initialize the deflector
        """
        if self.type == 'stdout':
            sys.stdout = self.original_output
        elif self.type == 'stderr':
            sys.stderr = self.original_output
            Logger.reset_logger()
        else:
            raise ValueError(f"Invalid type: {self.type}")
        self.close()
        self._null_initialize()
        return self
            
    def write(self, text : str | Any) -> None:
        """Write to the catcher (if catching) and original output stream (if keep_original)"""
        if not self.is_catching or self.keep_original:
            self.original_write(text)
        if self.is_catching:
            self.catcher_write(text)
        self.flush()
        
    def flush(self) -> None:
        """Flush the original output stream and the catcher"""
        self.original_std.flush()
        if self.is_catching:
            self.catcher_flush()

    def close(self) -> None:
        """Close the catcher"""
        if hasattr(self.catcher, 'close'):
            try:
                getattr(self.catcher, 'close')()
            except Exception as e:
                Logger.error(f"Error closing catcher: {e}")
                raise e

class DeflectorGroup:
    def __init__(self , catchers : dict[type_of_std , type_of_catcher] | type_of_catcher , 
                 write_functions : dict[type_of_std , str] | str , 
                 keep_original : bool = True):

        stdout_catcher = catchers.get('stdout', None) if isinstance(catchers , dict) else catchers
        stderr_catcher = catchers.get('stderr', None) if isinstance(catchers , dict) else catchers

        stdout_writer = write_functions.get('stdout', 'write') if isinstance(write_functions , dict) else write_functions
        stderr_writer = write_functions.get('stderr', 'write') if isinstance(write_functions , dict) else write_functions

        self.stdout = OutputDeflector('stdout', stdout_catcher , keep_original , stdout_writer)
        self.stderr = OutputDeflector('stderr', stderr_catcher , keep_original , stderr_writer)

    def start_catching(self):
        self.stdout.start_catching()
        self.stderr.start_catching()
        return self

    def end_catching(self):
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
        self.deflectors = DeflectorGroup(self.catchers , self.write_functions , self.keep_original).start_catching()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the output catcher , end stdout and stderr redirection"""
        self.deflectors.end_catching()

    def write(self, text : str | Any):
        """Write to the output catcher"""
        ...
    
    def flush(self):
        """Flush the output catcher"""
        ...

    @abstractmethod
    def get_contents(self) -> Any:
        """Get the contents of the output catcher"""
        ...

    @property
    def contents(self) -> Any:
        """Get the contents of the output catcher"""
        return self.get_contents()

    @property
    def catchers(self) -> dict[type_of_std , type_of_catcher] | type_of_catcher | Any:
        """Get the catchers of the output catcher"""
        if not hasattr(self , '_catchers'):
            return self

    @catchers.setter
    def catchers(self , value : dict[type_of_std , type_of_catcher] | type_of_catcher | Any):
        """Set the catchers of the output catcher"""
        self._catchers = value

    @property
    def write_functions(self) -> dict[type_of_std , str] | str:
        """Get the deflectors of the output catcher"""
        if not hasattr(self , '_write_functions'):
            self._write_functions = 'write'
        return self._write_functions
    
    @write_functions.setter
    def write_functions(self , value : dict[type_of_std , str] | str):
        """Set the write functions of the output catcher"""
        self._write_functions = value

    @property
    def export_file_list(self) -> list[Path]:
        """Get the export file list, usually the default path and the user provided path"""
        if not hasattr(self , '_export_file_list'):
            self._export_file_list : list[Path] = []
        return self._export_file_list

    @property
    def enabled(self) -> bool:
        """Check if the catcher is enabled"""
        if not hasattr(self , '_enable_catcher'):
            self._enable_catcher = True
        return self._enable_catcher

    @enabled.setter
    def enabled(self , value : bool):
        """Set the enabled status of the catcher"""
        self._enable_catcher = value

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
        self.catchers : dict[type_of_std , StringIO] = {'stdout': StringIO(), 'stderr': StringIO()}

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._contents = {key: catcher.getvalue() for key , catcher in self.catchers.items()}
        super().__exit__(exc_type, exc_val, exc_tb)
    
    def get_contents(self):
        return self._contents
    
    def clear(self):
        """Clear the contents of the output catcher"""
        for key , catcher in self.catchers.items():
            catcher.seek(0)
            catcher.truncate(0)

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
            self.log_file = None
        else:
            log_path = PATH.main.joinpath(log_path)
            log_path.parent.mkdir(exist_ok=True,parents=True)
            self.log_file = open(log_path, "w")
        self.catchers = {'stdout': self.log_file, 'stderr': self.log_file}

    def __enter__(self):
        super().__enter__()
        Proj.log_file = self.log_file
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        Proj.log_file = None
        super().__exit__(exc_type, exc_val, exc_tb)

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
    def __init__(self , catch_warnings : list[str] | None = None):
        self.warnings_caught = []
        self.original_showwarning = warnings.showwarning
        warnings.filterwarnings('always')
        self.catch_warnings = [] if catch_warnings is None else [c.lower() for c in catch_warnings]
    
    def custom_showwarning(self, message, category, filename, lineno, file=None, line=None) -> None:
        """Custom warning show function to catch specific warnings and show call stack"""
        # only catch the warnings we care about
        if any(c in str(message).lower() for c in self.catch_warnings):
            stack = traceback.extract_stack()
            Logger.alert1(f"\n caught warning: {message}")
            Logger.alert1(f"warning location: {filename}:{lineno}")
            Logger.alert1("call stack:")
            for i, frame in enumerate(stack[:-1]):  # exclude current frame
                Logger.alert1(f"  {i+1}. {frame.filename}:{frame.lineno} in {frame.name}")
                Logger.alert1(f"     {frame.line}")
            Logger.alert1("-" * 80)
            
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
    vb_level: int | None = None
    
    def __post_init__(self):
        self._time = datetime.now()
        self.type_fmt = self.get_type_fmt()
        self.type_str = self.get_type_str()

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
        vb_level = Proj.States.current_vb_level
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

        return cls(output_type, content , infos , valid , vb_level)
    
    def equivalent(self, other: 'TimedOutput') -> bool:
        """
        Check if the output item is equivalent to the other item
        equivalent means the content is very similar , or the progress bar is the same
        """
        if self.type in ['data_frame' , 'figure']:
            return False
        return self.type == other.type and str(self.content) == str(other.content)
    
    def to_html(self , index: int = 0):
        """Convert the output item to html"""
        if self.content is None: 
            text = None
        elif self.type == 'data_frame':
            text = _dataframe_to_html(self.content)
        elif self.type == 'figure':
            text = _figure_to_html(self.content)
        else:
            text = _str_to_html(self.content)
        if text is None: 
            return None
        text = f"""
                <tr class="output-row">
                    <td class="index-cell">{index}</td>
                    <td class="type-cell {self.type_fmt}-type">{self.type_str}</td>
                    <td class="vb-level-cell {self.type_fmt}-bg">{self.vb_level_str}</td>
                    <td class="time-cell {self.type_fmt}-bg">{self.time_str}</td>
                    <td class="content-cell {self.type_fmt}-bg {self.type_fmt}-content">{text}</td>
                </tr>
"""
        return text

class HtmlCatcher(OutputCatcher):
    """
    Html catcher for stdout, stderr, dataframe, and image (use Logger.display to display), export to html file at exit
    example:
        catcher = HtmlCatcher()
        with catcher:
            Logger.stdout('This will be caught')
        contents = catcher.contents
    """
    ExportDIR = PATH.log_catcher.joinpath('html')
    export_suffix : str = '.html'

    Instance : 'HtmlCatcher | None' = None
    InstanceList : list['HtmlCatcher'] = []
    Capturing : bool = True

    def __init__(self, title: str | bool | None = None , category : str = 'miscelaneous', init_time: datetime | None = None , 
                 add_time_to_title: bool = True, **kwargs):
        self.enabled = not isinstance(title , bool) or title
        self.title = title if isinstance(title , str) else 'html_catcher'
        self.category = category
        assert self.category.replace(' ' , '_') != '_crash_protection' , f'self.category is not allowed to be _crash_protection'
        self.init_time = init_time if init_time else datetime.now()

        self.filename = f'{self.title.replace(' ' , '_')}.{self.init_time.strftime("%Y%m%d%H%M%S")}.html' if add_time_to_title else f'{self.title.replace(' ' , '_')}.html'
        self.add_export_file(self.ExportDIR.joinpath(self.category.replace(' ' , '_') , self.filename))
        
        self.outputs: list[TimedOutput] = []
        self.kwargs = kwargs
        self.InstanceList.append(self)

        self.write_functions = {'stdout': 'write_stdout', 'stderr': 'write_stderr'}
        
    def __bool__(self):
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}(title={self.title})"
    
    @property
    def is_running(self):
        """Check if the catcher is running"""
        return self.Instance is not None
    
    def set_attrs(self , title : str | None = None , category : str | None = None):
        """
        Set the attributes of the catcher even after initialization
        title : str , the title of the catcher
        category : str , the category of the catcher
        """
        instance = self.Instance if self.Instance is not None else self
        if title: 
            instance.title = title
        if category: 
            instance.category = category
        return self

    @classmethod
    def AddExportFile(cls , export_path : Path | str | None = None):
        """Add an path to the export file list"""
        if cls.Instance is not None:
            cls.Instance.add_export_file(export_path)

    @classmethod
    def SetAttrs(cls , title : str | None = None , category : str | None = None):
        """Set the attributes of the catcher, class level (set all catchers)"""
        if cls.Instance is not None:
            cls.Instance.set_attrs(title , category)

    @classmethod
    def SetInstance(cls , instance : 'HtmlCatcher'):
        """Set the instance of the catcher, if the catcher is already running, block the new instance"""
        if cls.Instance is not None:
            Logger.alert1(f"{cls.Instance} is already running, blocking {instance}")
            instance._enable_catcher = False
        else:
            cls.Instance = instance

    @classmethod
    def ClearInstance(cls , instance : 'HtmlCatcher'):
        """Clear the instance of the catcher if the catcher is the current instance"""
        if cls.Instance is instance:
            cls.Instance = None

    def __enter__(self):
        self.SetInstance(self)
 
        if not self.enabled: 
            return self
        self.start_time = datetime.now()
        self.deflectors = DeflectorGroup(self.catchers , self.write_functions , self.keep_original).start_catching()
        self.redirect_display_function()
        Logger.remark(f"{self} Capturing Start" , prefix = True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled: 
            return
        self.export()   
        self.deflectors.end_catching()
        self.restore_display_function()
        self.ClearInstance(self)

    def export(self , export_path: Path | None = None):
        """Export the catcher to all paths in the export file list"""
        self.add_export_file(export_path)
        for file in Proj.States.export_html_files:
            self.add_export_file(file)
        if not self.export_file_list:
            return

        # log first and then export
        Logger.remark(f"{self} Capturing Finished, cost {Duration(since = self.start_time)}" , prefix = True)
        for export_path in self.export_file_list:
            Logger.footnote(f"{self.__class__.__name__} result saved to {export_path}" , indent = 1)
        Proj.States.exit_files.append(self.export_file_list[0])
        
        html_content = self.generate_html()
        for export_path in self.export_file_list:
            export_path.parent.mkdir(exist_ok=True,parents=True)
            export_path.write_text(html_content, encoding='utf-8')
        
    def redirect_display_function(self):
        """redirect Logger.Display to catcher"""
        Logger.Display.set_callbacks([self.add_output , self.stop_capturing] , [self.start_capturing])

    def restore_display_function(self):
        """restore Logger.Display functions"""
        Logger.Display.reset_callbacks()
    
    def generate_html(self):
        """generate html file with time ordered outputs"""
        sorted_outputs = sorted(self.outputs, key=lambda x: x.sort_key)

        html_segments = []
        for i, output in enumerate(sorted_outputs):
            html_content = output.to_html(i)
            if html_content is None: 
                continue
            html_segments.append(html_content)

        return ''.join([self._html_head() , *html_segments , self._html_tail()])
 
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
        return cls

    @classmethod
    def start_capturing(cls , *args, **kwargs):
        """Start the capturing of the catcher , class level (start all catchers)"""
        cls.Capturing = True
        return cls

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
       
    def _html_head(self):
        """Generate the html head , including the styles of the html file, and basic information of the catcher"""
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
        
        head = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title.title()} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</title>
    <style>
        body {{
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            background-color: #1e1e1e;
            color: #d4d4d4;
            line-height: 1.;
            margin: 1px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .header {{
            background-color: #2d2d30;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #007acc;
            border-right: 4px solid #007acc;
        }}
        .footer {{
            color: #888;
            font-size: 11px;
            display: flex;
        }}
        .add-infos {{
            color: #888;
            font-size: 11px;
            display: flex;
        }}
        .add-title {{
            color: #007acc;
            font-size: 12px;
            font-weight: bold;
        }}
        .add-key {{
            width: {key_width}px;     
            text-align: left;
            flex-shrink: 0;
        }}
        .add-seperator {{
            margin: 0 0.5em;
            padding : 0 10px;
        }}
        .add-value {{
            text-align: left;
        }}
        .output-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background-color: #252526;
            border-radius: 5px;
            overflow: hidden;
            border-left: 4px solid gray;
            border-right: 4px solid gray;
        }}
        .table-header {{
            background-color: #2d2d30;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .table-header th {{
            padding: 12px 4px;
            font-weight: bold;
            border-top: 2px solid #3a3a3a;
            border-bottom: 2px solid #3a3a3a;
            color: #ffffff;
        }}
        .col-index {{
            text-align: center;
            width: 10px;
            min-width: 1px;
        }}
        .col-type {{
            border-left: 0.5px solid #3a3a3a;
            text-align: center;
            width: 50px;
            min-width: 50px;
        }}
        .col-vb-level {{
            border-left: 0.5px solid #3a3a3a;
            text-align: center;
            width: 10px;
            min-width: 1px;
        }}
        .col-time {{
            border-left: 0.5px solid #3a3a3a;
            text-align: center;
            width: 50px;
            min-width: 50px;
        }}
        .col-content {{
            border-left: 0.5px solid #3a3a3a;
            text-align: left;
            width: auto;
        }}
        .output-row td {{
            border-bottom: 0.5px solid #3a3a3a;
            transition: background-color 0.15s ease;
            font-weight: bold;
            vertical-align: top;
            font-size: 11px;
            padding: 1px 4px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }}
        .output-row:last-child td {{
            border-bottom: 2px solid #3a3a3a;
        }}
        .output-row:hover{{
            outline: 2px solid white;
            transition: all 0.25s ease;
        }}
        .output-row:hover td {{
            background-color: #3b3b3b !important;
        }}
        .index-cell {{
            text-align: center;
        }}
        .type-cell {{
            border-left: 0.5px solid #3a3a3a;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .vb-level-cell {{
            border-left: 0.5px solid #3a3a3a;
            text-align: center;
            color: #9ca3af;
        }}
        .time-cell {{
            border-left: 0.5px solid #3a3a3a;
            color: #9ca3af;
            white-space: nowrap;
        }}
        .content-cell {{
            border-left: 0.5px solid #3a3a3a;
            word-wrap: break-word;
            word-break: break-word;
            overflow-x: auto;
            max-width: 100%;
        }}
        .stdout-type {{
            color: #4dff8e;
            border-left: 3px solid #4dff8e;
            background-color: #00802f;
        }}
        .stdout-bg {{
            background-color: #031909;
        }}
        .stdout-content {{
            white-space: pre-wrap;
        }}

        .stderr-type {{
            color: #ff4d4d;
            border-left: 3px solid #ff4d4d;
            background-color: #800000;
        }}
        .stderr-bg {{
            background-color: #190303;
        }}
        .stderr-content {{
            white-space: pre-wrap;
        }}

        .dataframe-type {{
            color: #4d72ff;
            border-left: 3px solid #4d72ff;
            background-color: #001a80;
        }}
        .dataframe-bg {{
            background-color: #000a33;
        }}

        .image-type {{
            color: #ffb84d;
            border-left: 3px solid #ffb84d;
            background-color: #804c00;
        }}
        .image-bg {{
            background-color: #191103;
        }}

        .other-type {{
            color: #db4dff;
            border-left: 3px solid #db4dff;
            background-color: #660080;
        }}
        .other-bg {{
            background-color: #150319;
        }}
        .other-content {{
            white-space: pre-wrap;
        }}

        .dataframe table {{
            border-collapse: collapse;
            width: auto;
            max-width: 100%;
            font-size: 11px;
        }}
        .dataframe th, .dataframe td {{
            border: 1px solid #ddd;
            padding: 4px 6px;
            text-align: left;
        }}
        .dataframe th {{
            font-weight: bold;
        }}
        .dataframe thead th {{
            background-color: #808080;
            color: white;
            font-weight: bold;
        }}
        .df-text {{
            color: #333;
            margin: 5px 0;
            font-weight: bold;
            font-size: 12px;
        }}
        .df-fallback {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            padding: 8px;
            border-radius: 3px;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.title.title()}</h1>
            {infos_script}
            <br>
            {infos_outputs}
            <br>
        </div>
        <table class="output-table">
            <thead class="table-header">
                <tr>
                    <th class="col-index">Id</th>
                    <th class="col-type">Type</th>
                    <th class="col-vb-level">Vb</th>
                    <th class="col-time">Time</th>
                    <th class="col-content">Content</th>
                </tr>
            </thead>
            <tbody>
"""
        return head

    @staticmethod
    def _html_tail():
        """Generate the html tail , including the end of the html file"""
        tail = """
            </tbody>
        </table>
        <div class="footer">
            <p>End of Table</p>
        </div>
    </div>
</body>
</html>
"""
        return tail

class _markdown_writer:
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
        text = _str_to_html(text.strip('\n'))
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
    ExportDIR = PATH.log_catcher.joinpath('markdown')
    export_suffix : str = '.md'
    InstanceList : list['MarkdownCatcher'] = []

    def __init__(self, title: str | bool | None = None,
                 category : str = 'miscelaneous',
                 init_time: datetime | None = None,
                 add_time_to_title: bool = True,
                 to_share_folder: bool = False ,
                 seperating_by: Literal['min' , 'hour'  , 'day'] | None = 'min',
                 **kwargs):

        self.enabled = not isinstance(title , bool) or title
        self.title = title if isinstance(title , str) else 'markdown_catcher'
        self.category = category
        self.init_time = init_time if init_time else datetime.now()
        
        self.filename = f'{self.title.replace(' ' , '_')}.{self.init_time.strftime("%Y%m%d%H%M%S")}.md' if add_time_to_title else f'{self.title.replace(' ' , '_')}.md'
        self.add_export_file(self.ExportDIR.joinpath(self.category.replace(' ' , '_') , self.filename))
        if to_share_folder and (share_folder_path := MACHINE.share_folder_path()) is not None:
            self.add_export_file(share_folder_path.joinpath('markdown_catcher' , self.filename))
        
        self.kwargs = kwargs
        self.seperating_by = seperating_by
        
        self.is_catching = False
        self.InstanceList.append(self)

        self.write_functions = {'stdout': 'write_stdout', 'stderr': 'write_stderr'}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(title={self.title})'

    def __enter__(self):
        if not self.enabled or not self.export_file_list:
            return self  

        self.start_time = datetime.now()
        
        self.stats = {
            'stdout_lines' : 0,
            'stderr_lines' : 0,
        }
        
        self.open_markdown_file()
        self.deflectors = DeflectorGroup(self.catchers , self.write_functions , self.keep_original).start_catching()
        self.is_catching = True
        Logger.remark(f"{self} Capturing Start" , prefix = True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled or not self.is_catching or not self.export_file_list: 
            return
        self.close_markdown_file()
        self.deflectors.end_catching()
        self.export()
        self.is_catching = False
    
    def open_markdown_file(self , max_running_files : int = 5):
        """
        Open the running markdown file , Generate the markdown header , 
        including the title, start time, and basic information of the catcher
        """
        i = 0
        running_filename = self.export_file_list[-1].with_suffix('.running.md')
        while running_filename.exists():
            running_filename = self.export_file_list[-1].with_suffix(f'{i}.running.md')
            i += 1
            if i >= max_running_files:
                Logger.error(f"Too many running markdown files, max_running_files={max_running_files}")
                running_filename = self.export_file_list[-1].with_suffix('.running.md')
                break
        self.running_filename = running_filename
        self.running_filename.parent.mkdir(exist_ok=True,parents=True)
        self.markdown_file = open(self.running_filename, 'w', encoding='utf-8')
        self.markdown_writer = _markdown_writer(self.markdown_file, self.seperating_by)

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
        if not self.enabled or not self.export_file_list:
            self.markdown_file.close()
            return
        Logger.remark(f"{self} Capturing Finished, cost {Duration(since = self.start_time)}" , prefix = True)
        self.markdown_file.close()
        for filename in self.export_file_list:
            if filename.exists(): 
                filename.unlink()
            filename.parent.mkdir(exist_ok=True,parents=True)
            try:
                shutil.copy(self.running_filename, filename)
            except OSError as e:
                Logger.error(f"Failed to copy {self.running_filename} to {filename}: {e}")
            Logger.footnote(f"{self.__class__.__name__} result saved to {filename}" , indent = 1)
        for path in self.running_filename.parent.glob(f'{self.filename.removesuffix(".md")}.*running.md'):
            path.unlink()
    
    def write_stdout(self, text : str):
        """Write stdout to the markdown file"""
        if self.is_catching and text.strip():
            self.markdown_writer.write(text)
            self.stats['stdout_lines'] += 1

    def write_stderr(self, text : str):
        """Write stderr to the markdown file"""
        if self.is_catching and text.strip():
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
    ExportDIR = PATH.runtime.joinpath('crash_protector')
    InstanceList : list['CrashProtectorCatcher'] = []

    def __init__(self, task_id : str | None = None, init_time: datetime | None = None, 
                 seperating_by: Literal['min' , 'hour'  , 'day'] | None = 'min', **kwargs):
        self.enabled = task_id is not None
        self.task_id = task_id
        self.init_time = init_time if init_time else datetime.now()
        
        self.filename = self.ExportDIR / f'{self.init_time.strftime("%Y%m%d")}.{str(self.task_id).replace('/' , '_')}.md'
    
        self.kwargs = kwargs
        self.seperating_by = seperating_by
        
        self.is_catching = False
        self.InstanceList.append(self)

        self.write_functions = {'stdout': 'write_stdout', 'stderr': 'write_stderr'}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(task_id={self.task_id})'

    def __enter__(self):
        if not self.enabled:
            return self
        self.start_time = datetime.now()
        self.open_markdown_file()
        self.deflectors = DeflectorGroup(self.catchers , self.write_functions , self.keep_original).start_catching()
        self.is_catching = True
        Logger.remark(f"{self} Capturing Start" , prefix = True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        self.deflectors.end_catching()
        Logger.remark(f"{self} Capturing Finished, cost {Duration(since = self.start_time)}" , prefix = True)
        Logger.footnote(f"{self.__class__.__name__} file {self.filename} removed" , indent = 1)
        self.is_catching = False
    
    def open_markdown_file(self):
        """
        Open the running markdown file , Generate the markdown header , 
        including the title, start time, and basic information of the catcher
        """
        self.filename.parent.mkdir(exist_ok=True,parents=True)
        self.markdown_file = open(self.filename, 'w', encoding='utf-8')
        self.markdown_writer = _markdown_writer(self.markdown_file, self.seperating_by)
        self.markdown_writer.header(f'{self.task_id} initiated at {self.init_time}')
    
    def write_stdout(self, text):
        """Write stdout to the crash protector file"""
        if self.is_catching and text.strip():
            self.markdown_writer.write(text)
            
    def write_stderr(self, text):
        """Write stderr to the crash protector file"""
        if self.is_catching and text.strip():
            self.markdown_writer.write(text , stderr = True)

    def get_contents(self):
        return ''
        
    def close(self):
        self.markdown_file.close()
        self.filename.unlink(missing_ok=True)
