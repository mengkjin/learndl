import re
import cProfile
import traceback

import pandas as pd

from datetime import datetime
from typing import Any , Callable , Literal , Sequence

from src.proj.env import PATH
from src.proj.proj import Proj
from src.proj.abc import Duration , stdout , stderr , FormatStr

from .display import Display
from .logfile import LogFile

LOG_LEVEL_TYPE = Literal['remark' , 'highlight' , 'debug' , 'info' , 'warning' , 'error' , 'critical']
LOG_LEVELS : list[LOG_LEVEL_TYPE] = ['remark' , 'highlight' , 'info' , 'debug' , 'warning' , 'error' , 'critical']
LOG_PALETTE : dict[LOG_LEVEL_TYPE, dict[str , Any]] = {
    'remark' : {'color' : 'lightblue' , 'level_prefix' : {'level' : 'REMARK' , 'color' : 'white' , 'bg_color' : 'lightblue'} , 'bold' : True},
    'debug' : {'color' : 'gray' , 'level_prefix' : {'level' : 'DEBUG' , 'color' : 'white' , 'bg_color' : 'gray'} , 'bold' : True},
    'info' : {'color' : 'lightgreen' , 'level_prefix' : {'level' : 'INFO' , 'color' : 'black' , 'bg_color' : 'lightgreen'} , 'bold' : True},
    'highlight' : {'color' : 'lightcyan' , 'level_prefix' : {'level' : 'HIGHLIGHT' , 'color' : 'black' , 'bg_color' : 'lightcyan'} , 'bold' : True},
    'warning' : {'color' : 'lightyellow' , 'level_prefix' : {'level' : 'WARNING' , 'color' : 'black' , 'bg_color' : 'lightyellow'} , 'bold' : True},
    'error' : {'color' : 'lightred' , 'level_prefix' : {'level' : 'ERROR' , 'color' : 'white' , 'bg_color' : 'lightred'} , 'bold' : True},
    'critical' : {'color' : 'lightpurple' , 'level_prefix' : {'level' : 'CRITICAL' , 'color' : 'white' , 'bg_color' : 'lightpurple'} , 'bold' : True},
}

LOG_FILE = LogFile.initiate('main' , 'project' , rotate = True)
VB = Proj.vb

def new_stdout(*args , indent = 0 , color = None , vb_level : int = 1 , **kwargs):
    """
    custom stdout message
    kwargs:
        indent: add prefix '  --> ' before the message
        color , bg_color , bold: color the message
        sep , end , file , flush: same as stdout
    """
    with VB.WithVbLevel(vb_level):
        fstr = stdout(*args , indent = indent , color = color , write = not VB.ignore(vb_level), **kwargs)
    return fstr

def new_stderr(*args , indent = 0 , color = None , vb_level : int = 1 , **kwargs):
    """
    custom stdout message
    kwargs:
        indent: add prefix '  --> ' before the message
        color , bg_color , bold: color the message
        sep , end , file , flush: same as stdout
    """
    with VB.WithVbLevel(vb_level):
        fstr = stderr(*args , indent = indent , color = color , write = not VB.ignore(vb_level), **kwargs)
    LOG_FILE.write(fstr.unformatted())
    return fstr

def new_print_exc(e : Exception , color : str = 'lightred' , bold : bool = True) -> str:
    """Print the exception"""
    error_msg = ''.join(msg for msg in traceback.format_exception(type(e), e, e.__traceback__)).strip()
    new_stderr(error_msg , color = color , bold = bold)
    return error_msg

def new_print_traceback_stack(color : str = 'lightyellow' , bold : bool = True) -> str:
    """Print the traceback stack"""
    stack = traceback.extract_stack()
    stack_str = 'Traceback Stack:\n'
    for i, frame in enumerate(stack[:-1]):  # exclude current frame
        stack_str += f"  {i+1}. {frame.filename}:{frame.lineno} in {frame.name}\n"
        stack_str += f"     {frame.line}\n"
    new_stderr(stack_str , color = color , bold = bold)
    return stack_str

class Logger:
    """
    custom colored log , config at PATH.conf / 'setting' / 'logger.yaml'
    method include:
        stdout level:
            - stdout: custom stdout (standard printing method) , can use indent , color , bg_color , bold , sep , end , file , flush kwargs
            - divider: long line on stdout
            - success (lightgreen) , skipping (gray) , footnote (gray) , alert1 (lightyellow) , alert2 (lightred) , alert3 (purple)

        stderr level:
            - stderr: custom stderr (standard printing method) , can use indent , color , bg_color , bold , sep , end , file , flush kwargs
            - info (green) , debug (gray) , warning (yellow) , error (red) , critical (purple)

        stdout / stderr level (depends on the prefix):
            -highlight (lightcyan) , remark (lightblue)

        conclusions:
            - conclude(message , level = 'critical') to add the message to the conclusions for later use
            - draw_conclusions: wrap the conclusions: output to stdout , merge into a single string and clear them
            - get_conclusions(level = 'critical'): Get the conclusions of given level
        
        context manager:
            - Paragraph: level 1: warning , level 2: info , level 3: highlight , level 4: debug
            - Timer: Timer class for timing the code, show the time in the best way
            - Profiler: Profiler class for profiling the code, show the profile result in the best way
    """
    _instance : 'Logger | Any' = None
    _conclusions : dict[LOG_LEVEL_TYPE , list[str]] = {level : [] for level in LOG_LEVELS}

    def __new__(cls, *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def log_only(cls , *args , **kwargs):
        """dump to log writer with no display"""
        if Proj.log_writer:
            Proj.log_writer.write(' '.join([str(s) for s in args]) + '\n')

    @classmethod
    def stdout(cls , *args , indent = 0 , color = None , vb_level : int = 1 , **kwargs):
        """
        custom stdout message
        kwargs:
            indent: add prefix '  --> ' before the message
            color , bg_color , bold: color the message
            sep , end , file , flush: same as stdout
        """
        return new_stdout(*args , indent = indent , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def stdout_msgs(cls , msg_list : Sequence[tuple[int , str] | str] , 
                    title : str | None = None , title_kwargs : dict[str , Any] = {'bold' : True , 'color' : 'lightgreen'} , 
                    color = 'auto' , indent = 0 , vb_level : int = 1 , bold = True , italic = True , **kwargs):
        """
        custom stdout message of multiple messages, each message is a tuple of (indent , message) or a string
        kwargs:
            indent: add prefix '  --> ' before the message
            color , bg_color , bold: color the message
            sep , end , file , flush: same as stdout
        """
        def color_selector(color : str | None , indent : int):
            return (None if indent <= 1 else 'gray') if color is None or color == 'auto' else color

        if title:
            new_stdout(FormatStr(title , **title_kwargs).formatted() , vb_level = vb_level)
            add_indent = 1
        else:
            add_indent = 0
        
        msgs = [(indent + add_indent , msg) if isinstance(msg , str) else (msg[0] + add_indent , *msg[1:]) for msg in msg_list]
        msgs = [FormatStr(msg , indent = indent , color = color_selector(color , indent) , 
                          bold = bold , italic = italic , **kwargs).formatted() for indent , msg in msgs]
        new_stdout('\n'.join(msgs) , vb_level = vb_level)

    @classmethod
    def stdout_pairs(cls , pair_list : Sequence[tuple[int , str , Any] | tuple[str , Any]] | dict[str , Any] , color = None , 
                     title : str | None = None , 
                     title_kwargs : dict[str , Any] = {'bold' : True , 'color' : 'lightgreen'} , 
                     indent = 0 , vb_level : int = 1 , italic = True , min_key_len : int = -1 , **kwargs):
        """
        custom stdout message of multiple pairs, each pair is a tuple of (indent , key , value) or a tuple of (key , value)
        kwargs:
            indent: add prefix '  --> ' before the message
            color , bg_color , bold: color the message
            sep , end , file , flush: same as stdout
        """
        def color_selector(color : str | None , indent : int):
            return (None if indent <= 1 else 'gray') if color is None or color == 'auto' else color

        if title:
            new_stdout(FormatStr(title , **title_kwargs).formatted() , vb_level = vb_level)
            add_indent = 1
        else:
            add_indent = 0
        
        if isinstance(pair_list , dict):
            pair_list = list(pair_list.items())

        pairs = [(pair[0] + add_indent , *pair[1:]) if len(pair) == 3 else (indent + add_indent , *pair) for pair in pair_list]
        pairs = [(indent , f'{FormatStr.indent_str(indent)}{key}' , value) for indent , key , value in pairs]
        
        max_key_len = max([len(indented_key) for _ , indented_key , _ in pairs])
        max_key_len = max(max_key_len , min_key_len)
        min_indent = min([indent for indent , _ , _ in pairs])
        pairs = [FormatStr(f'{indented_key:{max_key_len + 2*(indent - min_indent)}s} : {value}' , color = color_selector(color , indent) , 
                          italic = italic , **kwargs).formatted() for indent , indented_key , value in pairs]
        new_stdout('\n'.join(pairs) , vb_level = vb_level)

    @classmethod
    def stderr(cls , *args , indent = 0 , color = None , vb_level : int = 1 , **kwargs):
        """
        custom stdout message
        kwargs:
            indent: add prefix '  --> ' before the message
            color , bg_color , bold: color the message
            sep , end , file , flush: same as stdout
        """
        return new_stderr(*args , indent = indent , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def caption(cls , *args , **kwargs):
        """custom gray stdout message for caption (e.g. table / figure title)"""
        new_stdout(*args , color = 'white' , bg_color = 'gray' , bold = True , **kwargs)

    @classmethod
    def footnote(cls , *args , vb_level : int = 3 , **kwargs):
        """custom gray stdout message for footnote (e.g. saved information)"""
        new_stdout(f'**{args[0]}' , *args[1:] , color = 'gray' , bold = True , italic = True , vb_level = vb_level , **kwargs)
        
    @classmethod
    def success(cls , *args , **kwargs):
        """custom green stdout message for success"""
        new_stdout('Success :' , *args , color = 'lightgreen' , **kwargs)
    
    @classmethod
    def skipping(cls , *args , **kwargs):
        """custom skipping message"""
        new_stdout('Skipping:' , *args , color = 'gray' , **kwargs)

    @classmethod
    def alert1(cls , *args , color = 'lightyellow' , vb_level : int = 0 , **kwargs):
        """
        custom stdout message with color for alert
        level: 1 for yellow (warning) , 2 for red (error) , 3 for purple (critical)
        """
        new_stdout('Caution :' , *args , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def alert2(cls , *args , color = 'lightred' , vb_level : int = 0 , **kwargs):
        """
        custom stdout message with color for alert
        level: 1 for yellow (warning) , 2 for red (error) , 3 for purple (critical)
        """
        new_stdout('RedAlert:' , *args , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def alert3(cls , *args , color = 'purple' , vb_level : int = 0 , **kwargs):
        """
        custom stdout message with color for alert
        level: 1 for yellow (warning) , 2 for red (error) , 3 for purple (critical)
        """
        new_stdout('Emergent:' , *args , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def note(cls , *args , color = 'lightblue' , vb_level : int = 1 , **kwargs):
        """
        custom lightblue stdout message for remark
        """
        new_stdout(*args , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def remark(cls , *args , indent : int = 0 , vb_level : int = 2 , **kwargs):
        """custom lightblue stderr"""
        new_stderr(*args , indent = indent , vb_level = vb_level , **(LOG_PALETTE['remark'] | kwargs))

    @classmethod
    def debug(cls , *args , indent : int = 0 , vb_level : int = 2 , **kwargs):
        """Debug level stderr"""
        new_stderr(*args , indent = indent , vb_level = vb_level , **(LOG_PALETTE['debug'] | kwargs))

    @classmethod
    def info(cls , *args , indent : int = 0 , vb_level : int = 1 , **kwargs):
        """Info level stderr"""
        new_stderr(*args , indent = indent , vb_level = vb_level , **(LOG_PALETTE['info'] | kwargs))

    @classmethod
    def highlight(cls , *args , indent : int = 0 , vb_level : int = 1 , **kwargs):
        """custom lightcyan colored Highlight level message"""
        new_stderr(*args , indent = indent , vb_level = vb_level , **(LOG_PALETTE['highlight'] | kwargs))

    @classmethod
    def warning(cls , *args , indent : int = 0 , vb_level : int = 1 , **kwargs):
        """Warning level stderr"""
        new_stderr(*args , indent = indent , vb_level = vb_level , **(LOG_PALETTE['warning'] | kwargs))

    @classmethod
    def error(cls , *args , indent : int = 0 , vb_level : int = 0 , **kwargs):
        """Error level stderr"""
        new_stderr(*args , indent = indent , vb_level = vb_level , **(LOG_PALETTE['error'] | kwargs))

    @classmethod
    def critical(cls , *args , indent : int = 0 , vb_level : int = 0 , **kwargs):
        """Critical level stderr"""
        new_stderr(*args , indent = indent , vb_level = vb_level , **(LOG_PALETTE['critical'] | kwargs))

    @classmethod
    def divider(cls , width : int = 140 , char : Literal['-' , '=' , '*'] = '-' , msg : str | None = None , 
                color : str | None = None , bold : bool = True , vb_level : int = 0):
        """Divider mesge , use stdout"""
        if msg is None:
            msg = char * width
        elif len(msg) < width:
            cleft = char * max(0 , (width - len(msg) - 2) // 2)
            cright = char * max(0 , width - len(msg) - 2 - len(cleft))
            msg = ' '.join([cleft , msg , cright])
        new_stdout(msg , color = color , bold = bold , vb_level = vb_level)

    @classmethod
    def conclude(cls , *args : str , level : LOG_LEVEL_TYPE = 'critical'):
        """Add the message to the conclusions for later use"""
        msg = ' '.join([str(s) for s in args])
        cls._conclusions[level].append(msg)

    @classmethod
    def draw_conclusions(cls , simplify_errors : bool = True) -> str:
        """wrap the conclusions: printout , merge into a single string and clear them"""
        conclusion_strs = []
        num_conclusions = sum([len(cls._conclusions[level]) for level in LOG_LEVELS])
        if num_conclusions == 0:
            return ''

        with cls.Paragraph('Final Conclusions' , 3):
            for level , palette in LOG_PALETTE.items():
                if not cls._conclusions[level]:
                    continue
                new_stdout(f'There are {len(cls._conclusions[level])} {level.upper()} Conclusions:' , color = palette['color'] , vb_level = 0)
                if simplify_errors and level == 'error':
                    new_stdout('Please refer to the error messages for details.' , indent = 1 , vb_level = 0)
                    conclusion_strs.append(f'{level.upper()}: Please refer to the error messages for details.')
                else:
                    for conclusion in cls._conclusions[level]:
                        new_stdout(conclusion , indent = 1 , vb_level = 0)
                        conclusion_strs.append(f'{level.upper()}: {conclusion}')
                cls._conclusions[level].clear()
        return '\n'.join(conclusion_strs)

    @classmethod
    def get_conclusions(cls , type : LOG_LEVEL_TYPE) -> list[str]:
        """Get the conclusions"""
        return cls._conclusions[type]

    @classmethod
    def print_exc(cls , e : Exception , color : str = 'lightred' , bold : bool = True):
        """Print the exception"""
        return new_print_exc(e , color = color , bold = bold)

    @classmethod
    def print_traceback_stack(cls , color : str = 'lightyellow' , bold : bool = True):
        """Print the exception stack"""
        return new_print_traceback_stack(color = color , bold = bold)

    @classmethod
    def test_logger(cls):
        import tqdm , pandas as pd , matplotlib.pyplot as plt
        with VB.WithVB(VB.max):
            with cls.Paragraph('ParagraphI' , 1):
                cls.stdout('This is a stdout message')
                cls.stderr('This is a stderr message')
                cls.success('conclusion success message' , vb_level = 5)
                cls.note('remark message' , vb_level = 12)
                cls.footnote('footnote message' , vb_level = 99)
                cls.alert1('alert1 message')
                cls.alert2('alert2 message')
                cls.alert3('alert3 message')
                cls.skipping('skipping message' , indent = 1)
                cls.caption('caption message')
                cls.divider()

            with cls.Paragraph('ParagraphII' , 2):
                for level in LOG_LEVELS:
                    getattr(cls, level)(f'This is a {level} message')

            with cls.Paragraph('ParagraphIII' , 3):
                for level in LOG_LEVELS:
                    cls.conclude(f'This is a {level} conclusion' , level = level)

            with cls.Timer('Timer'):
                df = pd.DataFrame({'a':[1,2,3,4,5,6,7,8,9,10],'b':[4,5,6,7,8,9,10,11,12,13]})
                cls.display(df)
                fig = plt.figure()
                plt.plot([1,2,3],[4,5,6])
                plt.close(fig)
                
            with cls.Paragraph('ParagraphIV' , 4):
                cls.display(fig)

            with cls.Profiler('Profiler'):
                for i in tqdm.tqdm(range(100) , desc='processing'):
                    pass

                Proj.print_disk_info()

            try:
                raise Exception('test exception')
            except Exception as e:
                cls.print_exc(e)
                cls.print_traceback_stack()
                cls.conclude(f'test exception: {e}' , level = 'error')

    @classmethod
    def display(cls , obj , caption : str | None = None , vb_level : int = 1):
        """
        display the object
        """
        if caption is not None:
            cls.caption(caption , vb_level = vb_level)
        Display(obj , vb_level = vb_level)

    @classmethod
    def set_display_callbacks(cls , callbacks_before : list[Callable] | None = None, callbacks_after : list[Callable] | None = None):
        """
        set the callbacks before and after the display
        example:
            Logger.set_display_callbacks(callback_before , callback_after)
            means:
                before the display, the callback_before will be called
                after the display, the callback_after will be called
        """
        Display.set_callbacks(callbacks_before , callbacks_after)

    @classmethod
    def reset_display_callbacks(cls):
        """
        reset the callbacks before and after the display
        """
        Display.reset_callbacks()

    class Timer:
        """Timer class for timing the code, show the time in the best way"""
        def __init__(self , *args , silent = False , indent = 0 , vb_level : int = 1 , enter_vb_level : int = VB.max): 
            self.key = '/'.join(args)
            self.silent = silent
            self.indent = indent
            self.vb_level = vb_level
            self.enter_vb_level = enter_vb_level
    
        def __enter__(self):
            self._init_time = datetime.now()
            if not self.silent: 
                new_stdout(self.enter_str , indent = self.indent , vb_level = self.enter_vb_level)
        def __exit__(self, type, value, trace):
            if not self.silent:
                new_stdout(self.exit_str , indent = self.indent , vb_level = self.vb_level)

        @property
        def enter_str(self):
            """Get the enter string"""
            return f'{self.key} start ... '
        @property
        def exit_str(self):
            """Get the exit string"""
            return f'{self.key} finished! Cost {Duration(since = self._init_time)}'

    class Paragraph:
        """
        Format Enclosed process to count time
        example:
            with Logger.ParagraphI('Process Name'):
                Logger.info('This is the enclosed process...')
        """
        VB_LEVEL = 1
        def __init__(self , title : str , level : Literal[1,2,3,4] , char : Literal['-' , '=' , '*'] = '*', vb_level : int = 0):
            self.title = title.title()
            self.level = level
            self.char : Literal['-' , '=' , '*'] = char
            match level:
                case 1:
                    self.color = 'lightyellow'
                    self.vb_level = max(vb_level , 1)
                case 2:
                    self.color = 'lightgreen'
                    self.vb_level = max(vb_level , 2)
                case 3:
                    self.color = 'lightcyan'
                    self.vb_level = max(vb_level , 2)
                case 4:
                    self.color = 'gray'
                    self.vb_level = max(vb_level , 2)
        def __enter__(self):
            self._init_time = datetime.now()
            self.write(f'{self.title} Start at {self._init_time.strftime("%Y-%m-%d %H:%M:%S")}')
        def __exit__(self, *args): 
            self._end_time = datetime.now()
            self.write(f'{self.title} Finish at {self._end_time.strftime("%Y-%m-%d %H:%M:%S")}, Cost {Duration(self._end_time - self._init_time)}')
        def write(self , message : str):
            Logger.divider(char = self.char , msg = message , color = self.color , bold = True , vb_level = self.vb_level)
    
    class Profiler(cProfile.Profile):
        """Profiler class for profiling the code, show the profile result in the best way"""
        def __init__(self, title : str | None = None , builtins = False , display = True , n_head = 20 , 
                     columns = ['type' , 'name' , 'ncalls', 'cumtime' ,  'tottime' , 'percall' , 'where'] , 
                     sort_on = 'cumtime' , highlight = None ,
                     **kwargs) -> None:
            self.profiling = title is not None
            self.title = title
            if self.profiling: 
                super().__init__(builtins = builtins) 
            self.display = display
            self.n_head = n_head
            self.columns = columns
            self.sort_on = sort_on
            self.highlight = highlight

        def __enter__(self):
            if self.profiling: 
                self.start_time = datetime.now()
                return super().__enter__()
            else:
                return self

        def __exit__(self, type , value , trace):
            if type is not None:
                Logger.error(f'Error in Profiler ' , type , value)
                new_print_exc(value)
            elif self.profiling:
                if self.display:
                    new_stdout(f'Profiler cost time: {Duration(datetime.now() - self.start_time)}')
                    df = self.get_df(sort_on = self.sort_on , highlight = self.highlight)
                    Display(df.loc[:,self.columns].head(self.n_head))
                return super().__exit__(type , value , trace)

        def get_df(self , sort_on = 'cumtime' , highlight = None):
            """Get the profile result as a pandas dataframe"""
            if not self.profiling: 
                return pd.DataFrame()
            # highlight : 'gp_math_func.py'
            df = pd.DataFrame(
                getattr(self , 'getstats')(), columns=['full_name', 'ncalls', 'ccalls', 'cumtime' ,  'tottime' , 'caller']).\
                astype({'full_name':str})
            df['tottime'] = df['tottime'].round(4)
            df['cumtime'] = df['cumtime'].round(4)
            df_func = pd.DataFrame(
                [self.decompose_func_str(s) for s in df.full_name] , 
                columns = pd.Index(['type' , 'name' , 'where' , 'memory']))
            df = pd.concat([df_func , df],axis=1).sort_values(sort_on,ascending=False)
            df['percall'] = df['cumtime'] / df['ncalls']
            column_order = ['type' , 'name' , 'ncalls', 'ccalls', 'cumtime' ,  'tottime' , 'percall' , 'where' , 'memory' , 'full_name', 'caller']
            df = df.loc[:,column_order]
            if self.title is not None: 
                path = PATH.logs.joinpath('profiler' , f'{self.title.replace(" ","_")}.csv')
                df.to_csv(path)
                Logger.footnote(f'Profile result saved to {path}')
            if isinstance(highlight , str): 
                df = df[df.full_name.str.find(highlight) > 0]
            return df.reset_index(drop=True)

        @staticmethod
        def decompose_func_str(func_string):
            """
            Decompose the function string into a list of information
            return: type , name , where , memory
            """
            pattern = {
                r'<code object (.+) at (.+), file (.+), line (\d+)>' : ['function' , (0,) , (2,3) , (1,)] ,
                r'<function (.+) at (.+)>' : ['function' , (0,) , () , ()] ,
                r'<method (.+) of (.+) objects>' : ['method' , (0,) , (1,) , ()] ,
                r'<built-in method (.+)>' : ['built-in-method' , (0,) , () , ()] ,
                r'<fastparquet.(.+)>' : ['fastparquet' , () , () , ()] ,
                r'<pandas._libs.(.+)>' : ['pandas._libs' , (0,) , () , ()],
            }
            data = None
            for pat , use in pattern.items():
                match = re.match(pat, func_string)
                if match:
                    data = [use[0] , ','.join(match.group(i+1) for i in use[1]) , 
                            ','.join(match.group(i+1) for i in use[2]) , ','.join(match.group(i+1) for i in use[3])]
                    #try:
                    #    data = [use[0] , ','.join(match.group(i+1) for i in use[1]) , ','.join(match.group(i+1) for i in use[2])]
                    #except:
                    #    Logger.error(func_string)
                    break
            if data is None: 
                Logger.warning(f'Failed to decompose function string: {func_string}')
                data = [''] * 4
            data[2] = data[2].replace(str(PATH.main) , '')
            return data
