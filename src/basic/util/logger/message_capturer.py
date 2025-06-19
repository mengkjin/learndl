import io , sys , re , html , time , base64 , platform , socket
import pandas as pd
            
from dataclasses import dataclass , field
from datetime import datetime
from matplotlib.figure import Figure
from typing import Any , Callable
from pathlib import Path
from src.basic import path as PATH

class_mapping = {
    'stdout' : 'stdout',
    'stderr' : 'stderr',
    'data_frame' : 'dataframe',
    'plot' : 'image',
}

def critical(message: str):
    sys.stderr.write(f"\033[41m\033[1m{message}\033[0m\n")

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

class TeeOutput:
    """double output stream: output to console and recorder"""
    def __init__(self, original_stream, output_recorder : 'MessageCapturer' , output_type):
        self.original = original_stream
        self.recorder = output_recorder
        self.output_type = output_type
        
    def write(self, text : str | Any):
        # output to console
        self.original.write(text)
        self.original.flush()
        # record to time ordered list
        if text := text.strip():  # only record non-empty content
            self.recorder.add_output(self.output_type, text)
        
    def flush(self):
        self.original.flush()
    
    def __getattr__(self, name):
        return getattr(self.original, name)
    
class NoCapture:
    def __enter__(self):
        MessageCapturer.Capturing = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        MessageCapturer.Capturing = True
    
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
        elif self.type == 'plot':
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
    def create(cls, output_type: str, content: str | pd.DataFrame | pd.Series | Figure | None | Any):
        infos = {}
        valid = True
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
        elif self.type == 'plot':
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

class MessageCapturer:
    ExportDIR = PATH.log_record.joinpath('message_capturer')
    Instance : 'MessageCapturer | None' = None
    InstanceList : list['MessageCapturer'] = []
    Capturing : bool = True

    '''capture message from stdout and stderr'''
    def __init__(self, title: str  | None = None, export_path: Path | str | None = None , **kwargs):
        if title is None: title = 'message_capturer'
        self._enable_capturer = True
        self.outputs: list[TimedOutput] = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.original_display = {}
        self.title = title if title else 'message_capturer'
        self._export_path = Path(export_path) if isinstance(export_path ,  str) else export_path
        self.kwargs = kwargs
        self.__class__.InstanceList.append(self)
        
    def __bool__(self):
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}(title={self.title})"
    
    @property
    def enabled(self):
        return self._enable_capturer
    
    @property
    def is_running(self):
        return self.Instance is not None
    
    @classmethod
    def CreateCapturer(cls, message_capturer : Path | str | bool = False , title : str | None = None , **kwargs):
        export_path = message_capturer if isinstance(message_capturer , (Path , str)) else None
        capturer = cls(title , export_path = export_path , **kwargs)
        if not message_capturer: capturer._enable_capturer = False
        return capturer
    
    def get_export_path(self):
        if isinstance(self._export_path, Path):
            path = self._export_path
        else:
            path = self.ExportDIR.joinpath(f'{self.title.replace(" " , "_")}.{datetime.now().strftime("%Y%m%d%H%M%S")}.html')
        assert not path or path.suffix == '.html' , f"export_path must be a html file , but got {path}"
        return path
    
    def set_attrs(self , title : str | None = None , export_path : Path | str | None = None):
        if self.Instance is None or self.Instance is self:
            self.title = title if title else 'message_capturer'
            self._export_path = Path(export_path) if isinstance(export_path ,  str) else export_path
            assert not self._export_path or self._export_path.suffix == '.html' , f"export_path must be a html file , but got {self._export_path}"
        else:
            self.Instance.set_attrs(title , export_path)
        return self

    def SetInstance(self):
        if self.Instance is not None:
            critical(f"{self.Instance} is already running, blocking {self}")
            self._enable_capturer = False
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
        
        critical(f"{self} start to capture messages at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.redirect_display_functions()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled: return
        self.export()   
        self.restore_display_functions()

        self.ClearInstance()

    def export(self , export_path: Path | None = None):
        if export_path is None: export_path = self.get_export_path()
        critical(f"{self} Finished Capturing, saved to {export_path}")
        export_path.parent.mkdir(exist_ok=True,parents=True)
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_html())

        from src.basic.util.email import Email
        Email.attach(export_path)
        
    def redirect_display_functions(self):
        """redirect stdout, stderr, and display_module functions to capturer"""
        sys.stdout = TeeOutput(self.original_stdout, self, 'stdout')
        sys.stderr = TeeOutput(self.original_stderr, self, 'stderr')

        self.display_original = {}
        self.display_captured = {}
        try:
            import src.func.display as display_module
        except ImportError as e:
            critical(f"Cannot Import src.func.display: {e}")
            return
        else:
            for display_type in ['data_frame' , 'plot']:
                original = getattr(display_module , display_type)
                captured = self.display_wrapper(display_type , original)
                self.display_original[display_type] = original
                self.display_captured[display_type] = captured
                setattr(display_module , display_type , captured)

    def restore_display_functions(self):
        """restore stdout, stderr, and display_module functions"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        import src.func.display as display_module
        for key, value in self.original_display.items():
            setattr(display_module , key, value)

    def display_wrapper(self, display_type: str, original_func: Callable):
        assert display_type in ['data_frame' , 'plot'] , f"Unknown display function: {display_type}"        
        def wrapper(obj , *args, **kwargs):
            self.add_output(display_type, obj)
            with NoCapture():
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
 
    def add_output(self, type: str, content: str | pd.DataFrame | pd.Series | Figure):
        """add output to time ordered list"""
        if not self.Capturing: return
        output = TimedOutput.create(type , content)
        if not self.outputs or (output and not output.equivalent(self.outputs[-1])): 
            self.outputs.append(output)
       
    def _html_head(self):
        key_width = 80
        if self.kwargs:
            key_width = max(int(max(len(key) for key in list(self.kwargs.keys())) * 5.5) + 10 , key_width)
        
        infos = {
            'Machine' : socket.gethostname().split('.')[0],
            'Python' : f"{platform.python_version()}-{platform.machine()}",
            'Start at' : self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Finished at' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
    
    