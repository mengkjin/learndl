import colorlog , io , logging , sys , re , html , time , base64 , platform , socket
import pandas as pd
            
from dataclasses import dataclass
from datetime import datetime
from matplotlib.figure import Figure
from typing import Any , Callable
from pathlib import Path

from src.basic import path as PATH
from src.basic import conf as CONF

class Logger:
    '''custom colored log , config at {PATH.conf}/logger.yaml '''
    _instance = None
    def __new__(cls, *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , test_output = False):
        config_logger = CONF.glob('logger')
        new_path = PATH.log_main.joinpath(config_logger['file']['param']['filename'])
        config_logger['file']['param']['filename'] = str(new_path)
        new_path.parent.mkdir(exist_ok=True)
        log = logging.getLogger(config_logger['name'])
        exec("log.setLevel(logging."+config_logger['level']+")")

        while log.handlers:
            log.handlers[-1].close()
            log.removeHandler(log.handlers[-1])

        for hdname in config_logger['handlers']:
            exec(hdname+"_hdargs=config_logger[hdname]['param']")
            exec(hdname+"_handler="+config_logger[hdname]['class']+"(**"+hdname+"_hdargs)")
            exec(hdname+"_fmtargs=config_logger['formatters'][config_logger[hdname]['formatter']]")
            exec(hdname+"_formatter="+config_logger[hdname]['formatter_class']+"(datefmt=config_logger['datefmt'],**"+hdname+"_fmtargs)")
            exec(hdname+"_handler.setLevel(logging."+config_logger[hdname]['level']+")")
            exec(hdname+"_handler.setFormatter("+hdname+"_formatter)")
            exec("log.addHandler("+hdname+"_handler)")
        
        if test_output:
            log.debug('This is the DEBUG    message...')
            log.info('This is the INFO     message...')
            log.warning('This is the WARNING  message...')
            log.error('This is the ERROR    message...')
            log.critical('This is the CRITICAL message...')

        self.log = log

    def debug(self , *args , **kwargs):
        self.log.debug(*args , **kwargs)
        self.additional_writer(*args)
    
    def info(self , *args , **kwargs):
        self.log.info(*args , **kwargs)
        self.additional_writer(*args)

    def warning(self , *args , **kwargs):
        self.log.warning(*args , **kwargs)
        self.additional_writer(*args)

    def error(self , *args , **kwargs):
        self.log.error(*args , **kwargs)
        self.additional_writer(*args)   

    def critical(self , *args , **kwargs):
        self.log.critical(*args , **kwargs)
        self.additional_writer(*args)
        
    def additional_writer(self , *args):
        if isinstance(sys.stdout , DualPrinter):
            sys.stdout.log.write(' '.join([str(s) for s in args]) + '\n')


class _LevelFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, level_fmts=None):
        level_fmts = level_fmts or {}
        self._level_formatters = {}
        for level, format in level_fmts.items():
            # Could optionally support level names too
            self._level_formatters[getattr(logging , level)] = logging.Formatter(fmt=format, datefmt=datefmt)
        # self._fmt will be the default format
        super(_LevelFormatter, self).__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        if record.levelno in self._level_formatters:
            return self._level_formatters[record.levelno].format(record)
        return super(_LevelFormatter, self).format(record)
class _LevelColorFormatter(colorlog.ColoredFormatter):
    def __init__(self, fmt=None, datefmt=None, log_colors=None,level_fmts=None,secondary_log_colors=None):
        level_fmts = level_fmts or {}
        self._level_formatters = {}
        for level, format in level_fmts.items():
            # Could optionally support level names too
            self._level_formatters[getattr(logging , level)] = colorlog.ColoredFormatter(fmt=format, datefmt=datefmt , log_colors=log_colors , secondary_log_colors=secondary_log_colors)
        # self._fmt will be the default format
        super(_LevelColorFormatter, self).__init__(fmt=fmt, datefmt=datefmt,
                                                   log_colors=log_colors or {},
                                                   secondary_log_colors=secondary_log_colors or {})

    def format(self, record):
        if record.levelno in self._level_formatters:
            return self._level_formatters[record.levelno].format(record)
        return super(_LevelColorFormatter, self).format(record)

class DualPrinter:
    '''change print target to both terminal and file'''
    def __init__(self, filename : str):
        self.filename = PATH.log_update.joinpath(filename)
        self.filename.parent.mkdir(exist_ok=True,parents=True)
        self.terminal = sys.stdout
        self.log = open(self.filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for Python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        pass

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        self.log.close()

    def contents(self):
        with open(self.filename , 'r') as f:
            return f.read()

class MessageCapturer:
    '''capture message from stdout and stderr'''

    class TeeOutput:
        """double output stream: output to console and recorder"""
        def __init__(self, original_stream, output_recorder, output_type):
            self.original = original_stream
            self.recorder = output_recorder
            self.output_type = output_type
            
        def write(self, text):
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
        
    @dataclass
    class TimedOutput:
        """time ordered output item"""
        type: str
        content: str | pd.DataFrame | pd.Series | Figure | None | Any
        count: int = 0
        
        def __post_init__(self):
            self._time = time.time()

        @property
        def create_time(self):
            return self._time

        @property
        def sort_key(self):
            return self._time

        @property
        def time_str(self) -> str:
            return datetime.fromtimestamp(self._time).strftime('%H:%M:%S.%f')[:-3]
        
    def __init__(self, title: str , export_path: Path | str | None = None , **kwargs):
        self.title = title
        self.outputs: list[MessageCapturer.TimedOutput] = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.original_display = {}
        self.set_export_path(export_path)
        self.output_count = 0
        self.kwargs = kwargs

    @property
    def export_path(self):
        return self._export_path
    
    def set_export_path(self, value: Path | str | None):
        if value is None:
            self._export_path = None
        else:
            path = Path(value)
            assert path.suffix == '.html' , f"export_path must be a html file , but got {path}"
            path.parent.mkdir(exist_ok=True,parents=True)
            self._export_path = path

    @staticmethod
    def critical(message: str):
        sys.stderr.write(f"\033[41m\033[1m{message}\033[0m\n")

    @staticmethod
    def print(message: str):
        sys.stdout.write(message)

    def __enter__(self):
        self.start_time = datetime.now()
        self.redirect_display_functions()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        final_output = f"MessageCapturer Finished Capturing {self.title.title()}"
        if self.export_path: final_output += f", saved to {self.export_path}"
        self.critical(final_output)
        self.export()    
        self.restore_display_functions()

    def export(self):
        if self.export_path is None: return
        self.html_content = self.generate_html()
            
        with open(self.export_path, 'w', encoding='utf-8') as f:
            f.write(self.html_content)
        
    def redirect_display_functions(self):
        """redirect stdout, stderr, and display_module functions to capturer"""
        sys.stdout = self.TeeOutput(self.original_stdout, self, 'stdout')
        sys.stderr = self.TeeOutput(self.original_stderr, self, 'stderr')

        self.display_original = {}
        self.display_captured = {}
        try:
            import src.func.display as display_module
        except ImportError as e:
            self.critical(f"Cannot Import src.func.display: {e}")
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
            return original_func(obj , *args, **kwargs)
        return wrapper
    
    def generate_html(self):
        """generate html file with time ordered outputs"""
        sorted_outputs = sorted(self.outputs, key=lambda x: x.sort_key)

        html_segments = []
        for i, output in enumerate(sorted_outputs):
            html_content = self.output_to_html(output)
            if html_content is None: continue
            html_segments.append(html_content)

        return ''.join([self._html_head() , *html_segments , self._html_tail()])
 
    def add_output(self, type: str, content: str | pd.DataFrame | pd.Series | Figure):
        """add output to time ordered list"""
        if self._skip_output(type , content): return
        self.output_count += 1
        self.outputs.append(self.TimedOutput(type, content , self.output_count))

    @classmethod
    def _skip_output(cls, type : str , content : str | Any) -> bool:
        if type == 'stderr':
            r0 = re.search(r"(\d{1,3})%\|", content) # XX%|
            r1 = re.search(r"(\d+)/(\d+)", content)  # XX/XX
            r2 = re.search(r"\[\d+:\d+<\d+:\d+,\s*([^\]]+)it/s]", content) # [XX:XX<XX:XX, XXXXit/s]
            if (r0 and int(r0.group(1)) != 100) and (r1 and int(r1.group(2)) != int(r1.group(1))) and r2: 
                return True
        return False

    @classmethod
    def output_to_html(cls, output: 'MessageCapturer.TimedOutput') -> str | None:
        if output.content is None: return None
        if output.type in ['stdout' , 'stderr']:
            assert isinstance(output.content, str) , f"output.content must be a string , but got {type(output.content)}"
            if (output.content == 'stderr' and 
                re.match(r"(?!100%\|)\d{1,3}%\|", output.content) and 
                output.content.rstrip().endswith("it/s]")): return None  # skip unfinished progress bar in stderr
            text = html.escape(output.content)
            text = re.sub(r'\u001b\[(\d+;)*\d+m', cls._convert_ansi_sequence, text)
            text = cls._ansi_to_css(text)
        elif output.type == 'data_frame':
            text = cls._dataframe_to_html(output.content)
        elif output.type == 'plot':
            text = cls._figure_to_html(output.content)
        else:
            raise ValueError(f"Unknown output type: {output.type}")
        if text is None: return None
        text = f"""
                <tr class="output-row">
                    <td class="index-cell">{output.count}</td>
                    <td class="type-cell {output.type}-type">{output.type}</td>
                    <td class="time-cell">{output.time_str}</td>
                    <td class="content-cell">
                        <div class="{output.type}-content">{text}</div>
                    </td>
                </tr>
"""
        return text
    
    @classmethod
    def _str_to_html(cls, text: str | Any):
        """capture string"""
        assert isinstance(text, str) , f"text must be a string , but got {type(text)}"
        if re.match(r"^(?!100%\|)\d{1,2}%\|", text): return None  # skip unfinished progress bar
        text = html.escape(text)
        text = re.sub(r'\u001b\[(\d+;)*\d+m', cls._convert_ansi_sequence, text)
        text = cls._ansi_to_css(text)
        return text

    @classmethod
    def _dataframe_to_html(cls, df: pd.DataFrame | pd.Series | Any):
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
        
    @classmethod
    def _figure_to_html(cls, fig: Figure | Any):
        """capture matplotlib figure"""
        assert isinstance(fig, Figure) , f"fig must be a matplotlib figure , but got {type(fig)}"
        content = None
        try:
            if fig.get_axes():  # check if figure has content
                if image_base64 := cls._figure_to_base64(fig):
                    content = f'<img src="data:image/png;base64,{image_base64}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0;">'
        except Exception as e:
            cls.critical(f"Error capturing matplotlib figure: {e}")
        return content
           
    def _html_head(self):
        key_width = max(int(max(len(key) for key in list(self.kwargs.keys())) * 5.5) + 10 , 80)
        
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
        .dataframe-type {{
            background-color: #1e3a8a;
            color: #60a5fa;
            border-left: 3px solid #3b82f6;
        }}
        .image-type {{
            background-color: #7c2d12;
            color: #fb923c;
            border-left: 3px solid #ea580c;
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
        .image-content {{
            padding: 1px 4px;
            border-radius: 1px;
            text-align: center;
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
    
    @staticmethod
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

    @classmethod
    def _figure_to_base64(cls, fig : Figure | Any):
        """convert matplotlib figure to base64 string"""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            return image_base64
        except Exception as e:
            cls.critical(f"\033[41m\033[1mError converting figure to base64: {e}\033[0m\n")
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
