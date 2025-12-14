import io , sys , re , html , base64 , platform , warnings , shutil , traceback
import pandas as pd

from typing import Any ,Literal , IO
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from dataclasses import dataclass , field
from datetime import datetime
from matplotlib.figure import Figure

from .path import PATH
from .logger import Logger
from .machine import MACHINE
from .timer import Duration
from .display import Display

__all__ = [
    'IOCatcher' , 'LogWriter' , 'OutputCatcher' , 'OutputDeflector' , 
    'HtmlCatcher' , 'MarkdownCatcher' , 'WarningCatcher' ,
]

class OutputDeflector:
    """
    double output stream: deflect output to catcher and original output stream (optional)
    example:
        catcher = IOCatcher()
        with OutputDeflector('stdout', catcher, keep_original=True):
            print('This will be deflected to catcher')
        with OutputDeflector('stderr', catcher, keep_original=False):
            Logger.info('This will be deflected to catcher')
    """
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
    
    def _null_initialize(self) -> None:
        """Null initialize the deflector"""
        self.original_output = None
        self.is_catching = False
        if hasattr(self, '_catcher_write'):
            del self._catcher_write
        if hasattr(self, '_catcher_flush'):
            del self._catcher_flush

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
                print(f"Error closing catcher: {e}")
                raise e

class OutputCatcher(ABC):
    """
    Abstract base class for output catcher for stdout and stderr
    must implement the get_contents method
    can rewrite the stdout_catcher/stderr_catcher or stdout_deflector/stderr_deflector
    """
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
    """
    Output catcher into StringIO for stdout and stderr
    example:
        catcher = IOCatcher()
        with catcher:
            print('This will be caught')
        contents = catcher.contents
    """
    keep_original = False
    
    def __init__(self):
        self.stdout_catcher = StringIO()
        self.stderr_catcher = StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._contents = {
            'stdout': self.stdout_catcher.getvalue(),
            'stderr': self.stderr_catcher.getvalue(),
        }
        super().__exit__(exc_type, exc_val, exc_tb)
    
    def get_contents(self):
        return self._contents
    
    def clear(self):
        self.stdout_catcher.seek(0)
        self.stdout_catcher.truncate(0)
        
        self.stderr_catcher.seek(0)
        self.stderr_catcher.truncate(0)

class LogWriter(OutputCatcher):
    """
    Output catcher into log file for stdout and stderr
    example:
        catcher = LogWriter('log.txt')
        with catcher:
            print('This will be caught')
        contents = catcher.contents
    """
    def __init__(self, log_path : str | Path | None = None):
        self.log_path = log_path
        if log_path is None: 
            self.log_file = None
        else:
            log_path = PATH.main.joinpath(log_path)
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

class WarningCatcher:
    """
    catch specific warnings and show call stack
    example:
        with WarningCatcher(['This will raise an exception']):
            print('This will raise an exception')
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
            print(f"\n caught warning: {message}")
            print(f"warning location: {filename}:{lineno}")
            print("call stack:")
            for i, frame in enumerate(stack[:-1]):  # exclude current frame
                print(f"  {i+1}. {frame.filename}:{frame.lineno} in {frame.name}")
                print(f"     {frame.line}")
            print("-" * 80)
            
            raise Exception(message)
        
        # call original warning show function
        self.original_showwarning(message, category, filename, lineno, file, line)
    
    def __enter__(self):
        warnings.showwarning = self.custom_showwarning
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.showwarning = self.original_showwarning

def _critical(message: str):
    purple_bg_white_bold = "\u001b[45m\u001b[1m\u001b[37m"  # purple backgroud , white bold
    reset_all = "\u001b[0m"
    # purple_text = "\u001b[35m\u001b[1m"  # purple text (no background) , bold

    level_name = 'CRITICAL'
    prefix = f'{datetime.now().strftime("%y-%m-%d %H:%M:%S")}|LEVEL:{level_name:9s}|'
    
    output = f"{purple_bg_white_bold}{prefix}{reset_all}: {purple_bg_white_bold}{message}{reset_all}\n"
    sys.stderr.write(output)
    return output

def _str_to_html(text: str | Any):
    """capture string"""
    
    assert isinstance(text, str) , f"text must be a string , but got {type(text)}"
    if re.match(r"^(?!100%\|)\d{1,2}%\|", text): 
        return None  # skip unfinished progress bar
    text = html.escape(text)
    text = re.sub(r'(?:\u001b\[[\d;]*m)+', replace_ansi_sequences, text)
    
    return text

def replace_ansi_sequences(match):
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
    
    for code_str in codes:
        if not code_str:
            continue
        code = int(code_str)
        if code == 0:
            return '</span>'
        elif code == 1:
            styles.append('font-weight: bold')
        elif 30 <= code <= 37:  # foreground color
            colors = ['black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white']
            fg_color = colors[code - 30]
        elif 40 <= code <= 47:  # background color
            colors = ['black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white']
            bg_color = colors[code - 40]
    
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
        _critical(f"Error converting figure to base64: {e}")
        return None
    
def _ansi_to_css(ansi_string: str) -> str:
    """convert ANSI color codes to CSS"""
    mapping =  {
        #'\u001b[0m': '</span>',  # reset
        '\u001b[0m': '</span>',  # reset
        '\u001b[1m': '<span style="font-weight: bold;">',  # bold
        '\u001b[31m': '<span style="color: red;">',  # red
        '\u001b[32m': '<span style="color: green;">',  # green
        '\u001b[33m': '<span style="color: yellow;">',  # yellow
        '\u001b[34m': '<span style="color: blue;">',  # blue
        '\u001b[35m': '<span style="color: purple;">',  # purple
        '\u001b[36m': '<span style="color: cyan;">',  # cyan
        '\u001b[37m': '<span style="color: white;">',  # white
        '\u001b[41m': '<span style="background-color: red; color: white;">',  # red background
        '\u001b[42m': '<span style="background-color: green; color: white;">',  # green background
        '\u001b[43m': '<span style="background-color: yellow; color: black;">',  # yellow background
        '\u001b[44m': '<span style="background-color: blue; color: white;">',  # blue background
        '\u001b[45m': '<span style="background-color: purple; color: white;">',  # purple background
        '\u001b[46m': '<span style="background-color: cyan; color: black;">',  # cyan background
        '\u001b[47m': '<span style="background-color: white; color: black;">',  # white background
        '\u001b[91m': '<span style="color: lightred">', # 亮红色
        '\u001b[92m': '<span style="color: lightgreen">', # 亮绿色
        '\u001b[93m': '<span style="color: lightyellow">', # 亮黄色
        '\u001b[94m': '<span style="color: lightblue">', # 亮蓝色
        '\u001b[95m': '<span style="color: lightpurple">', # 亮洋红色
        '\u001b[96m': '<span style="color: lightcyan">', # 亮青色
    }
    for ansi_code, html_code in mapping.items():
        ansi_string = ansi_string.replace(ansi_code, html_code)
    return ansi_string

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
                content = f'<img src="data:image/png;base64,{image_base64}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0;">'
    except Exception as e:
        _critical(f"Error capturing matplotlib figure: {e}")
    return content

@dataclass
class TimedOutput:
    """time ordered output item"""
    type: str
    content: str | pd.DataFrame | pd.Series | Figure | None | Any
    infos: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    
    def __post_init__(self):
        self._time = datetime.now()

    def __bool__(self):
        return self.valid
    
    @property
    def format_type(self):
        return {
            'stdout' : 'stdout',
            'stderr' : 'stderr',
            'data_frame' : 'dataframe',
            'figure' : 'image',
        }[self.type]
    
    @property
    def type_str(self):
        if self.type == 'stderr':
            return 'STDERR'
        elif self.type == 'stdout':
            return 'STDOUT'
        elif self.type == 'data_frame':
            return 'TABLE'
        elif self.type == 'figure':
            return 'IMAGE'

    @property
    def create_time(self):
        return self._time.timestamp()

    @property
    def sort_key(self):
        return self._time.timestamp()

    @property
    def time_str(self) -> str:
        return self._time.strftime('%H:%M:%S.%f')[:-3]
    
    @classmethod
    def create(cls, content: str | pd.DataFrame | pd.Series | Figure | None | Any , output_type: str | None = None):
        infos = {}
        valid = True
        if output_type is None:
            if isinstance(content , Figure): 
                output_type = 'figure'
            elif isinstance(content , pd.DataFrame): 
                output_type = 'data_frame'
            else: 
                raise ValueError(f"Unknown output type: {type(content)}")
        if output_type == 'stderr':
            assert isinstance(content, str) , f"content must be a string , but got {type(content)}"
            r0 = re.search(r"^(.*?)(\d{1,3})%\|", content) # capture text before XX%|
            r1 = re.search(r"(\d+)/(\d+)", content)  # XX/XX
            r2 = re.search(r"\s*([^\]]+),\s*([^\]]+)it/s]", content) # [XXXXXXX, XXXXit/s]
            if r0 and r1 and r2: # is a progress bar
                infos['is_progress_bar'] = True
                infos['unique_content'] = str(r0.group(1))
                if int(r0.group(2)) != 100 or (int(r1.group(1)) != int(r1.group(2))): # not finished
                    valid = False
        elif output_type == 'stdout':
            assert isinstance(content, str) , f"content must be a string , but got {type(content)}"
            if content == '...': 
                valid = False

        return cls(output_type, content , infos , valid)
    
    def equivalent(self, other: 'TimedOutput') -> bool:
        if self.type in ['data_frame' , 'figure']:
            return False
        if self.type == other.type:
            if str(self.content) == str(other.content): 
                return True
            elif self.type == 'stderr':
                if self.infos.get('is_progress_bar' , False) and other.infos.get('is_progress_bar' , False):
                    uc0 = self.infos.get('unique_content' , '')
                    uc1 = other.infos.get('unique_content' , '')
                    return (uc0 == uc1) and (uc0 != '')
        return False
    
    def to_html(self , index: int = 0):
        if self.content is None: 
            return None
        if self.type in ['stdout' , 'stderr']:
            text = _str_to_html(self.content)
        elif self.type == 'data_frame':
            text = _dataframe_to_html(self.content)
        elif self.type == 'figure':
            text = _figure_to_html(self.content)
        else:
            raise ValueError(f"Unknown output type: {self.type}")
        if text is None: 
            return None
        text = f"""
                <tr class="output-row">
                    <td class="index-cell">{index}</td>
                    <td class="type-cell {self.format_type}-type">{self.type_str}</td>
                    <td class="time-cell">{self.time_str}</td>
                    <td class="content-cell">
                        <div class="{self.format_type}-content">{text}</div>
                    </td>
                </tr>
"""
        return text

class HtmlCatcher(OutputCatcher):
    ExportDIR = PATH.log_catcher.joinpath('html')
    Instance : 'HtmlCatcher | None' = None
    InstanceList : list['HtmlCatcher'] = []
    Capturing : bool = True

    '''catch message from stdout and stderr, and display module'''
    def __init__(self, title: str | bool | None = None , category : str = 'miscelaneous', init_time: datetime | None = None , 
                 add_time_to_title: bool = True, **kwargs):
        if isinstance(title , bool) and not title:
            self._enable_catcher = False
        else:
            self._enable_catcher = True
        
        self.title = title if isinstance(title , str) else 'html_catcher'
        self.category = category
        self.init_time = init_time if init_time else datetime.now()

        self.filename = f'{self.title.replace(' ' , '_')}.{self.init_time.strftime("%Y%m%d%H%M%S")}.html' if add_time_to_title else f'{self.title.replace(' ' , '_')}.html'
        self.add_export_file(self.ExportDIR.joinpath(self.category.replace(' ' , '_') , self.filename))

        self.outputs: list[TimedOutput] = []
        self.kwargs = kwargs
        self.InstanceList.append(self)
        
    def __bool__(self):
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}(title={self.title})"
    
    @property
    def enabled(self):
        return self._enable_catcher
    
    @property
    def is_running(self):
        return self.Instance is not None

    @property
    def export_file_list(self) -> list[Path]:
        if not hasattr(self , '_export_file_list'):
            self._export_file_list : list[Path] = []
        return self._export_file_list

    def add_export_file(self , export_path : Path | str | None = None):
        if export_path is None:
            return
        export_path = Path(export_path) if isinstance(export_path ,  str) else export_path
        assert export_path.suffix == '.html' , f"export_path must be a html file , but got {export_path}"
        self._export_file_list.append(export_path)
    
    def set_attrs(self , title : str | None = None , export_path : Path | str | None = None , category : str | None = None):
        instance = self.Instance if self.Instance is not None else self
        if title: 
            instance.title = title
        if category: 
            instance.category = category
        if export_path: 
            instance.add_export_file(export_path)
        return self

    def SetInstance(self):
        if self.Instance is not None:
            _critical(f"{self.Instance} is already running, blocking {self}")
            self._enable_catcher = False
        else:
            self.__class__.Instance = self

        return self

    def ClearInstance(self):
        if self.Instance is self:
            self.__class__.Instance = None
        return self

    def __enter__(self):
        self.SetInstance()
 
        if not self.enabled: 
            return self
        self.start_time = datetime.now()
        self.redirect_display_function()
        _critical(f"{self} Capturing Start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled: 
            return
        self.export()   
        self.restore_display_function()
        self.ClearInstance()

    def export(self , export_path: Path | None = None , add_to_email: bool = True):
        self.add_export_file(export_path)
        if not self.export_file_list:
            return
        html_content = self.generate_html()
        _critical(f"{self} Capturing Finished, cost {Duration(since = self.start_time)}")
        for export_path in self.export_file_list:
            export_path.parent.mkdir(exist_ok=True,parents=True)
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            _critical(f"{self.__class__.__name__} result saved to {export_path}")
        
        if add_to_email:
            from src.basic.util.email import Email
            Email.Attach(self.export_file_list[-1])
        
    def redirect_display_function(self):
        """redirect stdout, stderr, and proj.display.Display to catcher"""
        self.stdout_deflector = OutputDeflector('stdout', self , self.keep_original , 'write_stdout')
        self.stderr_deflector = OutputDeflector('stderr', self , self.keep_original , 'write_stderr')
        self.stdout_deflector.start_catching()
        self.stderr_deflector.start_catching()

        Display.set_callbacks([self.add_output , self.stop_capturing] , [self.start_capturing])

    def restore_display_function(self):
        """restore stdout, stderr, and display_module functions"""
        self.stdout_deflector.end_catching()
        self.stderr_deflector.end_catching()
        Display.reset_callbacks()
    
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
        if not self.outputs or (output and not output.equivalent(self.outputs[-1])): 
            self.outputs.append(output)

    @classmethod
    def stop_capturing(cls , *args, **kwargs):
        cls.Capturing = False
        return cls

    @classmethod
    def start_capturing(cls , *args, **kwargs):
        cls.Capturing = True
        return cls

    def write_stdout(self, text: str):
        if text := text.strip():
            self.add_output(text, 'stdout')

    def write_stderr(self, text: str):
        if text := text.strip():
            self.add_output(text, 'stderr')

    def get_contents(self):
        return self.generate_html()
       
    def _html_head(self):
        key_width = 80
        if self.kwargs:
            key_width = max(int(max(len(key) for key in list(self.kwargs.keys())) * 5.5) + 10 , key_width)
        finish_time = datetime.now()
        infos = {
            'Machine' : MACHINE.name,
            'Python' : f"{platform.python_version()}-{platform.machine()}",
            'Start at' : f'{self.start_time.strftime("%Y-%m-%d %H:%M:%S")}',
            'Finish at' : f'{finish_time.strftime("%Y-%m-%d %H:%M:%S")}',
            'Duration' : Duration((finish_time - self.start_time).total_seconds()).fmtstr,
            'Outputs Num' : len(self.outputs)
        }
        infos_block = '\n'.join([f'<div class="add-infos"><span class="add-key">{key}</span><span class="add-seperator">:</span><span class="add-value">{value}</span></div>' 
                                 for key, value in infos.items()])
        kwargs_block = '\n'.join([f'<div class="add-kwargs"><span class="add-key">{key}</span><span class="add-seperator">=</span><span class="add-value">{value}</span></div>' 
                                  for key, value in self.kwargs.items()])
        
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
        }}
        .add-kwargs {{
            color: #007acc;
            font-size: 11px;
            display: flex;
        }}
        .add-infos {{
            color: #888;
            font-size: 11px;
            display: flex;
        }}
        .add-title {{
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
            border: 1px solid #3e3e42;
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
            border-bottom: 2px solid #007acc;
            color: #ffffff;
        }}
        .col-index {{
            text-align: center;
            width: 10px;
            min-width: 1px;
        }}
        .col-type {{
            text-align: center;
            width: 50px;
            min-width: 50px;
        }}
        .col-time {{
            text-align: center;
            width: 50px;
            min-width: 50px;
        }}
        .col-content {{
            text-align: left;
            width: auto;
        }}
        .output-row {{
            border-bottom: 1px solid #3e3e42;
        }}
        .output-row:hover {{
            background-color: #2d2d30;
        }}
        .output-row:last-child {{
            border-bottom: none;
        }}
        .index-cell {{
            padding: 1px 4px;
            font-weight: bold;
            text-align: center;
            vertical-align: top;
            font-size: 11px;
        }}
        .type-cell {{
            padding: 1px 4px;
            font-weight: bold;
            text-align: center;
            vertical-align: top;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stdout-type {{
            background-color: #1a472a;
            color: #4ade80;
            border-left: 3px solid #22c55e;
        }}
        .stderr-type {{
            background-color: #7f1d1d;
            color: #f87171;
            border-left: 3px solid #ef4444;
        }}
        .time-cell {{
            padding: 1px 4px;
            font-size: 11px;
            color: #9ca3af;
            vertical-align: top;
            white-space: nowrap;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }}
        .content-cell {{
            padding: 1px 4px;
            vertical-align: top;
            word-wrap: break-word;
            word-break: break-word;
        }}
        .stdout-content {{
            background-color: #1e1e1e;
            color: #d4d4d4;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
        }}
        .stderr-content {{
            background-color: #2d1b1b;
            color: #ffcccc;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
        }}
        .dataframe-type {{
            background-color: #1e3a8a;
            color: #60a5fa;
            border-left: 3px solid #3b82f6;
        }}
        .dataframe-content {{
            padding: 1px 4px;
            border-radius: 1px;
            overflow-x: auto;
            max-width: 100%;
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
        .image-type {{
            background-color: #7c2d12;
            color: #fb923c;
            border-left: 3px solid #ea580c;
        }}
        .image-content {{
            padding: 1px 4px;
            border-radius: 1px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.title.title()}</h1>
            <div class="add-infos"><span class="add-title"> INFORMATION </span></div>
            {infos_block}
            <div class="add-kwargs"><span class="add-title"> KEYWORD ARGUMENTS </span></div>
            {kwargs_block}
        </div>
        <table class="output-table">
            <thead class="table-header">
                <tr>
                    <th class="col-index">Id</th>
                    <th class="col-type">Type</th>
                    <th class="col-time">Time</th>
                    <th class="col-content">Content</th>
                </tr>
            </thead>
            <tbody>
"""
        return head

    @staticmethod
    def _html_tail():
        tail = """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return tail
    
class MarkdownCatcher(OutputCatcher):
    ExportDIR = PATH.log_catcher.joinpath('markdown')
    InstanceList : list['OutputCatcher'] = []

    def __init__(self, title: str | bool | None = None,
                 category : str = 'miscelaneous',
                 init_time: datetime | None = None,
                 add_time_to_title: bool = True,
                 to_share_folder: bool = False ,
                 seperating_by: Literal['min' , 'hour'  , 'day'] | None = 'min',
                 **kwargs):

        if isinstance(title , bool) and not title:
            self._enable_catcher = False
        else:
            self._enable_catcher = True

        self.title = title if isinstance(title , str) else 'markdown_catcher'
        self.category = category
        self.init_time = init_time if init_time else datetime.now()
        
        self.filename = f'{self.title.replace(' ' , '_')}.{self.init_time.strftime("%Y%m%d%H%M%S")}.md' if add_time_to_title else f'{self.title.replace(' ' , '_')}.md'
        self.add_export_file(self.ExportDIR.joinpath(self.category.replace(' ' , '_') , self.filename))
        if to_share_folder and (share_folder_path := MACHINE.share_folder_path()) is not None:
            self.add_export_file(share_folder_path.joinpath('markdown_catcher' , self.filename))
        
        
        self.kwargs = kwargs
        self.last_seperator = None
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
        self.is_catching = False

        self.InstanceList.append(self)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(title={self.title})'

    @property
    def export_file_list(self) -> list[Path]:
        if not hasattr(self , '_export_file_list'):
            self._export_file_list : list[Path] = []
        return self._export_file_list

    @property
    def enabled(self) -> bool:
        return self._enable_catcher

    def add_export_file(self , export_path : Path | str | None = None):
        if export_path is None:
            return
        export_path = Path(export_path) if isinstance(export_path ,  str) else export_path
        assert export_path.suffix == '.md' , f"export_path must be a markdown file , but got {export_path}"
        self._export_file_list.append(export_path)

    def __enter__(self):
        if not self.enabled or not self.export_file_list:
            return self  

        self.start_time = datetime.now()
        
        self.stats = {
            'stdout_lines' : 0,
            'stderr_lines' : 0,
        }
        
        self._open_markdown_file()
        self._markdown_header()
        self.stdout_deflector = OutputDeflector('stdout', self, True , 'write_stdout').start_catching()
        self.stderr_deflector = OutputDeflector('stderr', self, True , 'write_stderr').start_catching()
        self.is_catching = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled or not self.is_catching or not self.export_file_list: 
            return
        self._markdown_footer()
        self.markdown_file.flush()
        self.stdout_deflector.end_catching()
        self.stderr_deflector.end_catching()
        self.export()
        self.is_catching = False
        return False
    
    def _open_markdown_file(self):
        i = 0
        running_filename = self.export_file_list[-1].with_suffix('.running.md')
        while running_filename.exists():
            running_filename = running_filename.with_suffix(f'.{i}.md')
        self.running_filename = running_filename
        self.running_filename.parent.mkdir(exist_ok=True,parents=True)
        self.markdown_file = open(self.running_filename, 'w', encoding='utf-8')
        
    def export(self):
        self.markdown_file.close()
        if not self.enabled or not self.export_file_list:
            return
        _critical(f"{self} Capturing Finished, cost {Duration(since = self.start_time)}")
        for filename in self.export_file_list:
            if filename.exists(): 
                filename.unlink()
            filename.parent.mkdir(exist_ok=True,parents=True)
            shutil.copy(self.running_filename, filename)
            _critical(f"{self.__class__.__name__} result saved to {filename}")
        self.running_filename.unlink()
    
    def _markdown_header(self):
        self.markdown_file.write(f"# {self.title.title()}\n")
        self.markdown_file.write(f"## Log Start \n")
        self.markdown_file.write(f"- *Machine: {MACHINE.name}*  \n")
        self.markdown_file.write(f"- *Python: {platform.python_version()}-{platform.machine()}*  \n")
        self.markdown_file.write(f"- *Start at: {self.start_time}*  \n")
        self.markdown_file.write(f"## Log Main \n")

    def _markdown_footer(self):
        finish_time = datetime.now()
        self.markdown_file.write(f"## Log End \n")
        self.markdown_file.write(f"- *Finish at: {finish_time}*  \n")
        self.markdown_file.write(f"- *Duration: {Duration((finish_time - self.start_time).total_seconds()).fmtstr}*  \n")
        self.markdown_file.write(f"- *Stdout Lines: {self.stats['stdout_lines']}*  \n")
        self.markdown_file.write(f"- *Stderr Lines: {self.stats['stderr_lines']}*  \n")
        self.markdown_file.write(f"***\n")
        self.markdown_file.flush()

    def _markdown_seperator(self):
        if self.seperating_by is None: 
            return
        seperator = self._seperator_time_str(self.last_time)
        if seperator != self.last_seperator:
            self.markdown_file.write(f"### {seperator} \n")
            self.last_seperator = seperator
            self.markdown_file.flush()
    
    def formatted_text(self, text : str , prefix='- ' , suffix='  \n' , type: Literal['stdout' , 'stderr'] = 'stdout'):
        """replace ANSI color codes with HTML span tags"""
        text = text.strip('\n')
        text = _ansi_to_css(text)
        if type == 'stderr':
            text = f'<u>{text}</u>'
        text = prefix + text + suffix
        return text
    
    def write_std(self , text: str , type: Literal['stdout' , 'stderr']):
        if self.is_catching and (text := text.strip()):
            self.last_time = datetime.now()
            self.stats[f'{type}_lines'] += 1
            text = f'{self._prefix_time_str(self.last_time)} {text}'
            text = self.formatted_text(text , type = type)
            self._markdown_seperator()
            self.markdown_file.write(text)
            self.markdown_file.flush()
    
    def write_stdout(self, text):
        self.write_std(text , 'stdout')

    def write_stderr(self, text):
        self.write_std(text , 'stderr')

    def get_contents(self):
        if self.running_filename.exists():
            with open(self.running_filename , 'r') as f:
                return f.read()
        elif self.export_file_list:
            with open(self.export_file_list[-1] , 'r') as f:
                return f.read()
        else:
            return ''
        
    def close(self):
        self.markdown_file.close()
