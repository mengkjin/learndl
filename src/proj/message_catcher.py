import io , sys , re , html , time , base64 , platform 
import pandas as pd
            
from dataclasses import dataclass , field
from datetime import datetime , timedelta
from matplotlib.figure import Figure
from typing import Any , Callable , Literal
from pathlib import Path

from .machine import MACHINE
from .path import PATH
from .output_catcher import OutputDeflector , OutputCatcher

__all__ = ['HtmlCatcher' , 'MarkdownCatcher']

class_mapping = {
    'stdout' : 'stdout',
    'stderr' : 'stderr',
    'data_frame' : 'dataframe',
    'figure' : 'image',
}

def critical(message: str):
    sys.stderr.write(f"\u001b[41m\u001b[1m{message}\u001b[0m\n")

def _str_to_html(text: str | Any):
    """capture string"""
    assert isinstance(text, str) , f"text must be a string , but got {type(text)}"
    if re.match(r"^(?!100%\|)\d{1,2}%\|", text): return None  # skip unfinished progress bar
    text = html.escape(text)
    text = re.sub(r'\u001b\[(\d+;)*\d+m', _convert_ansi_sequence, text)
    text = _ansi_to_css(text)
    return text

def _convert_ansi_sequence(match):
    """convert complex ANSI sequence"""
    sequence = match.group(0)
    codes = sequence[2:-1].split(';')  # remove \u001b[ and m
    
    styles = []
    bg_color = None
    fg_color = None
    
    for code in codes:
        if not code: continue
        code = int(code)
        if code == 0:
            return '</span>'
        elif code == 1:
            styles.append('font-weight: bold')
        elif 30 <= code <= 37:  # foreground color
            colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
            fg_color = colors[code - 30]
        elif 40 <= code <= 47:  # background color
            colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
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
        critical(f"Error converting figure to base64: {e}")
        return None
    
@staticmethod
def _ansi_to_css(ansi_string: str) -> str:
    mapping =  {
        #'\u001b[0m': '</span>',  # reset
        '\u001b[0m': '</span>',  # reset
        '\u001b[1m': '<span style="font-weight: bold;">',  # bold
        '\u001b[31m': '<span style="color: red;">',  # red
        '\u001b[32m': '<span style="color: green;">',  # green
        '\u001b[33m': '<span style="color: yellow;">',  # yellow
        '\u001b[34m': '<span style="color: blue;">',  # blue
        '\u001b[35m': '<span style="color: magenta;">',  # magenta
        '\u001b[36m': '<span style="color: cyan;">',  # cyan
        '\u001b[37m': '<span style="color: white;">',  # white
        '\u001b[41m': '<span style="background-color: red; color: white;">',  # red background
        '\u001b[42m': '<span style="background-color: green; color: white;">',  # green background
        '\u001b[43m': '<span style="background-color: yellow; color: black;">',  # yellow background
        '\u001b[44m': '<span style="background-color: blue; color: white;">',  # blue background
        '\u001b[45m': '<span style="background-color: magenta; color: white;">',  # magenta background
        '\u001b[46m': '<span style="background-color: cyan; color: black;">',  # cyan background
        '\u001b[47m': '<span style="background-color: white; color: black;">',  # white background
        '\u001b[91m': '<span style="color: lightred">', # 亮红色
        '\u001b[92m': '<span style="color: lightgreen">', # 亮绿色
        '\u001b[93m': '<span style="color: lightyellow">', # 亮黄色
        '\u001b[94m': '<span style="color: lightblue">', # 亮蓝色
        '\u001b[95m': '<span style="color: lightmagenta">', # 亮洋红色
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
    except Exception as e:
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
        critical(f"Error capturing matplotlib figure: {e}")
    return content

def _strf_delta(tdelta : timedelta):
    """
    Custom function to format a timedelta object.
    
    Args:
        tdelta (timedelta): The timedelta object to format.
        fmt (str): The format string, using placeholders like {D}, {H}, {M}, {S}.
    """
    # Extract total seconds, including days.
    assert tdelta.seconds >= 0 , f"tdelta must be a positive timedelta , but got {tdelta}"
    total_seconds = int(tdelta.total_seconds())

    # Handle negative timedeltas
    total_seconds = abs(total_seconds)

    # Calculate time components
    days, remainder = divmod(total_seconds, 86400) # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)    # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)      # 60 seconds in a minute
    
    # Store components in a dictionary for f-string formatting
    fmtstr = ''
    if days:
        fmtstr += f'{days} Days '
    if hours:
        fmtstr += f'{hours} Hours '
    if minutes:
        fmtstr += f'{minutes} Minutes '
    if seconds:
        fmtstr += f'{seconds} Seconds'
    if not fmtstr:
        fmtstr = '<1 Seconds'
    return fmtstr

@dataclass
class TimedOutput:
    """time ordered output item"""
    type: str
    content: str | pd.DataFrame | pd.Series | Figure | None | Any
    infos: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    
    def __post_init__(self):
        self._time = time.time()

    def __bool__(self):
        return self.valid
    
    @property
    def format_type(self):
        return class_mapping[self.type]
    
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
        return self._time

    @property
    def sort_key(self):
        return self._time

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self._time).strftime('%H:%M:%S.%f')[:-3]
    
    @classmethod
    def create(cls, content: str | pd.DataFrame | pd.Series | Figure | None | Any , output_type: str | None = None):
        infos = {}
        valid = True
        if output_type is None:
            if isinstance(content , Figure): output_type = 'figure'
            elif isinstance(content , pd.DataFrame): output_type = 'data_frame'
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
            if content == '...': valid = False

        return cls(output_type, content , infos , valid)
    
    def equivalent(self, other: 'TimedOutput') -> bool:
        if self.type == other.type:
            if self.content == other.content: 
                return True
            elif self.type == 'stderr':
                if self.infos.get('is_progress_bar' , False) and other.infos.get('is_progress_bar' , False):
                    uc0 = self.infos.get('unique_content' , '')
                    uc1 = other.infos.get('unique_content' , '')
                    return (uc0 == uc1) and (uc0 != '')
        return False
    
    def to_html(self , index: int = 0):
        if self.content is None: return None
        if self.type in ['stdout' , 'stderr']:
            text = _str_to_html(self.content)
        elif self.type == 'data_frame':
            text = _dataframe_to_html(self.content)
        elif self.type == 'figure':
            text = _figure_to_html(self.content)
        else:
            raise ValueError(f"Unknown output type: {self.type}")
        if text is None: return None
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
    ExportDIR = PATH.log_autorun.joinpath('html_catcher')
    Instance : 'HtmlCatcher | None' = None
    InstanceList : list['HtmlCatcher'] = []
    Capturing : bool = True
    DisplayModule : Any = None

    class NoCapture:
        def __enter__(self):
            HtmlCatcher.Capturing = False
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            HtmlCatcher.Capturing = True

    '''catch message from stdout and stderr, and display module'''
    def __init__(self, title: str  | None = None, export_path: Path | str | None = None , time: datetime | None = None , **kwargs):
        self._enable_catcher = True
        self.outputs: list[TimedOutput] = []
        self.title = title if title else 'html_catcher'
        self.time = time if time else datetime.now()
        self._export_path = Path(export_path) if isinstance(export_path ,  str) else export_path
        self.kwargs = kwargs
        self.__class__.InstanceList.append(self)

        try:
            import src.func.display as display_module
            self.__class__.DisplayModule = display_module
        except ImportError as e:
            raise ImportError(f"Cannot Import src.func.display: {e}")
        
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
    
    @classmethod
    def CreateCatcher(cls, message_catcher : Path | str | bool = False , title : str | None = None , time : datetime | None = None , **kwargs):
        export_path = message_catcher if isinstance(message_catcher , (Path , str)) else None
        catcher = cls(title , export_path = export_path , time = time , **kwargs)
        if not message_catcher: catcher._enable_catcher = False
        return catcher
    
    def get_export_path(self):
        if isinstance(self._export_path, Path):
            path = self._export_path
        else:
            title = self.title.replace(" " , "_")
            time_str = self.time.strftime('%Y%m%d%H%M%S')
            path = self.ExportDIR.joinpath(title , f'{title}.{time_str}.html')
        assert not path or path.suffix == '.html' , f"export_path must be a html file , but got {path}"
        return path
    
    def set_attrs(self , title : str | None = None , export_path : Path | str | None = None , time : datetime | None = None):
        if self.Instance is None or self.Instance is self:
            if title: self.title = title
            if time: self.time = time
            if export_path: self._export_path = Path(export_path) if isinstance(export_path ,  str) else export_path
            assert not self._export_path or self._export_path.suffix == '.html' , f"export_path must be a html file , but got {self._export_path}"
        else:
            self.Instance.set_attrs(title , export_path)
        return self

    def SetInstance(self):
        if self.Instance is not None:
            critical(f"{self.Instance} is already running, blocking {self}")
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
 
        if not self.enabled: return self
        self.start_time = datetime.now()
        self.redirect_display_function()
        critical(f"{self} start to capture messages at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled: return
        self.export()   
        self.restore_display_function()

        self.ClearInstance()

    def export(self , export_path: Path | None = None):
        if export_path is None: export_path = self.get_export_path()
        critical(f"{self} Finished Capturing, saved to {export_path}")
        export_path.parent.mkdir(exist_ok=True,parents=True)
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_html())
        
        from src.basic.util.email import Email
        Email.Attach(export_path)
        
    def redirect_display_function(self):
        """redirect stdout, stderr, and display_module functions to catcher"""
        self.stdout_deflector = OutputDeflector('stdout', self , self.keep_original , 'write_stdout')
        self.stderr_deflector = OutputDeflector('stderr', self , self.keep_original , 'write_stderr')
        self.stdout_deflector.start_catching()
        self.stderr_deflector.start_catching()

        self.original_display = self.DisplayModule.display
        self.DisplayModule.display = self.display_wrapper(self.DisplayModule.display)

    def restore_display_function(self):
        """restore stdout, stderr, and display_module functions"""
        self.stdout_deflector.end_catching()
        self.stderr_deflector.end_catching()
        self.DisplayModule.display = self.original_display

    def display_wrapper(self, original_func: Callable):
        def wrapper(obj , *args, **kwargs):
            self.add_output(obj)
            with self.NoCapture():
                display_result = original_func(obj , *args, **kwargs)
            return display_result
        return wrapper
    
    def generate_html(self):
        """generate html file with time ordered outputs"""
        sorted_outputs = sorted(self.outputs, key=lambda x: x.sort_key)

        html_segments = []
        for i, output in enumerate(sorted_outputs):
            html_content = output.to_html(i)
            if html_content is None: continue
            html_segments.append(html_content)

        return ''.join([self._html_head() , *html_segments , self._html_tail()])
 
    def add_output(self, content: str | pd.DataFrame | pd.Series | Figure | Any , output_type: str | None = None):
        """add output to time ordered list"""
        if not self.Capturing: return
        
        output = TimedOutput.create(content , output_type)
        if not self.outputs or (output and not output.equivalent(self.outputs[-1])): 
            self.outputs.append(output)

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
        duration = _strf_delta(finish_time - self.start_time)
        infos = {
            'Machine' : MACHINE.name,
            'Python' : f"{platform.python_version()}-{platform.machine()}",
            'Start at' : self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Finish at' : finish_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Duration' : duration,
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
    <title>{self.title.title()} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
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
    ExportDIR = PATH.log_autorun.joinpath('markdown_catcher')
    ExportDIR.mkdir(exist_ok=True,parents=True)

    def __init__(self, title: str  | None = None, 
                 to_share_folder: bool = False ,
                 seperating_by: Literal['min' , 'hour'  , 'day'] | None = 'min',
                 add_time_to_title: bool = True,
                 given_export_path: Path | str | None = None , 
                 **kwargs):
        self.title = title if title else 'markdown_catcher'
        self.start_time = datetime.now()
        if given_export_path is None:
            if add_time_to_title:
                filename = f'{self.title}.{self.start_time.strftime("%Y%m%d%H%M%S")}.md'
            else:
                filename = f'{self.title}.md'
            self.filename = self.ExportDIR.joinpath(filename)
        else:
            self.filename = Path(given_export_path)
        if to_share_folder:
            if MACHINE.share_folder_path is None:
                self.filename = None
            else:
                self.filename = MACHINE.share_folder_path.joinpath('markdown_catcher' , self.filename.name)
        
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

    def __enter__(self):
        if self.filename is None or not self.filename.parent.exists():
            return self
        
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
        if not self.is_catching or self.filename is None: return
        self._markdown_footer()
        self.markdown_file.flush()
        self.stdout_deflector.end_catching()
        self.stderr_deflector.end_catching()
        self._close_markdown_file()
        print(f"Markdown saved to {self.filename.absolute()}")
        return False
    
    def _open_markdown_file(self):
        assert self.filename is not None and self.filename.parent.exists() , f"filename must be a valid file path"
        i = 0
        running_filename = self.filename.with_suffix('.running.md')
        while running_filename.exists():
            running_filename = running_filename.with_suffix(f'.{i}.md')
        self.running_filename = running_filename
        self.markdown_file = open(self.running_filename, 'w', encoding='utf-8')
        
    def _close_markdown_file(self):
        self.markdown_file.close()
        if self.filename is not None:
            if self.filename.exists(): self.filename.unlink()
            self.running_filename.rename(self.filename)
        if self.running_filename.exists(): 
            self.running_filename.unlink()
    
    def _markdown_header(self):
        self.markdown_file.write(f"# {self.title.title()}\n")
        self.markdown_file.write(f"## Log Start \n")
        self.markdown_file.write(f"- *Machine: {MACHINE.name}*  \n")
        self.markdown_file.write(f"- *Python: {platform.python_version()}-{platform.machine()}*  \n")
        self.markdown_file.write(f"- *Start at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}*  \n")
        self.markdown_file.write(f"## Log Main \n")

    def _markdown_footer(self):
        finish_time = datetime.now()
        duration = _strf_delta(finish_time - self.start_time)
        self.markdown_file.write(f"## Log End \n")
        self.markdown_file.write(f"- *Finish at: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}*  \n")
        self.markdown_file.write(f"- *Duration: {duration}*  \n")
        self.markdown_file.write(f"- *Stdout Lines: {self.stats['stdout_lines']}*  \n")
        self.markdown_file.write(f"- *Stderr Lines: {self.stats['stderr_lines']}*  \n")
        self.markdown_file.write(f"***\n")
        self.markdown_file.flush()

    def _markdown_seperator(self):
        if self.seperating_by is None: return
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
        if self.filename is None or not self.filename.exists(): return ''
        with open(self.filename , 'r') as f:
            return f.read()
        
    def close(self):
        self.markdown_file.close()
