import colorlog , logging
import logging.handlers

import re
import cProfile
import traceback

import pandas as pd

from datetime import datetime
from typing import Any , Literal , Type

from src.proj.env import PATH , ProjStates , ProjConfig
from src.proj.func import Duration , Display , stdout , stderr , FormatStr

class _LevelFormatter(logging.Formatter):
    """Simple Level Formatter without color"""
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
    """Level Color Formatter with default colors"""
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

def log_config() -> dict[str, Any]:
    """Initialize the logger"""
    log_config = PATH.read_yaml(PATH.conf.joinpath('glob' , 'logger.yaml'))
    new_path = PATH.log_main.joinpath(log_config['file']['param']['filename'])
    log_config['file']['param']['filename'] = str(new_path)
    new_path.parent.mkdir(exist_ok=True)

    return log_config

def new_log(config : dict[str, Any]) -> logging.Logger:
    """create a new logger with the config"""
    log = logging.getLogger(config['name'])
    log.setLevel(getattr(logging , config['level']))
    reset_logger(log , config)
    return log

def reset_logger(log : logging.Logger , config : dict[str, Any]):
    """reset the log writer"""
    while log.handlers:
        log.handlers[-1].close()
        log.removeHandler(log.handlers[-1])

    for hdname in config['handlers']:
        hdargs = config[hdname]['param']
        hdclass : Type[logging.Handler] = eval(config[hdname]['class'])
        handler = hdclass(**hdargs)
        handler.setLevel(config[hdname]['level'])
        hdformatter = eval(config[hdname]['formatter_class'])(
            datefmt=config['datefmt'],
            **config['formatters'][config[hdname]['formatter']])
        handler.setFormatter(hdformatter)
        log.addHandler(handler)
    return log

class Logger:
    """
    custom colored log , config at PATH.conf / 'glob' / 'logger.yaml'
    method include:
        stdout: custom stdout (standard printing method) , can use indent , color , bg_color , bold , sep , end , file , flush kwargs
        success: custom stdout with green color
        remark: custom stdout , level [0,1,2,3] indicate the color of the message: 0 for blue (normal) , 1 for yellow (warning) , 2 for red (error) , 3 for purple (critical) (blue color stdout)
        skipping: custom skipping message (gray color stdout)
        highlight: custom cyan colored Highlight level message (cyan color stdout)
        info: Info level stderr
        debug: Debug level stderr
        warning: Warning level stderr
        error: Error level stderr
        critical: Critical level stderr
        divider: Divider stdout (use info level)
        conclude: Add the message to the conclusions for later use
        draw_conclusions: wrap the conclusions: printout , merge into a single string and clear them
        get_conclusions: Get the conclusions of given level
        ParagraphI: Format Enclosed process to count time
        ParagraphII: Format Enclosed process to count time
        ParagraphIII: Format enclosed message
        Profiler: Profiler class for profiling the code, show the profile result in the best way
    """
    _instance : 'Logger | Any' = None
    _type_levels = Literal['info' , 'debug' , 'warning' , 'error' , 'critical']
    _levels : list[_type_levels] = ['info' , 'debug' , 'warning' , 'error' , 'critical']
    _levels_palette : list[str] = ['lightgreen' , 'gray' , 'lightyellow' , 'lightred' , 'purple']
    _conclusions : dict[_type_levels , list[str]] = {level : [] for level in _levels}
    _config = log_config()
    log = new_log(_config)

    def __new__(cls, *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_logger(cls):
        """reset the log writer"""
        reset_logger(cls.log , cls._config)

    @classmethod
    def log_only(cls , *args , **kwargs):
        """dump to log writer with no display"""
        if log_file := ProjStates.log_file:
            log_file.write(' '.join([str(s) for s in args]) + '\n')

    @classmethod
    def stdout(cls , *args , indent = 0 , color = None , vb_level : int = 1 , **kwargs):
        """
        custom stdout message
        kwargs:
            indent: add prefix '  --> ' before the message
            color , bg_color , bold: color the message
            sep , end , file , flush: same as print
        """
        stdout(*args , indent = indent , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def caption(cls , *args , **kwargs):
        """custom gray stdout message for caption (e.g. table / figure title)"""
        stdout(*args , color = 'white' , bg_color = 'gray' , bold = True , **kwargs)

    @classmethod
    def footnote(cls , *args , **kwargs):
        """custom gray stdout message for footnote (e.g. saved information)"""
        stdout(*args , color = 'gray' , **kwargs)
        
    @classmethod
    def success(cls , *args , **kwargs):
        """custom green stdout message for success"""
        stdout(*args , color = 'lightgreen' , **kwargs)

    @classmethod
    def alert1(cls , *args , color = 'lightyellow' , vb_level : int = 0 , **kwargs):
        """
        custom stdout message with color for alert
        level: 1 for yellow (warning) , 2 for red (error) , 3 for purple (critical)
        """
        stdout(*args , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def alert2(cls , *args , color = 'lightred' , vb_level : int = 0 , **kwargs):
        """
        custom stdout message with color for alert
        level: 1 for yellow (warning) , 2 for red (error) , 3 for purple (critical)
        """
        stdout(*args , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def alert3(cls , *args , color = 'purple' , vb_level : int = 0 , **kwargs):
        """
        custom stdout message with color for alert
        level: 1 for yellow (warning) , 2 for red (error) , 3 for purple (critical)
        """
        stdout(*args , color = color , vb_level = vb_level , **kwargs)

    @classmethod
    def skipping(cls , *args , **kwargs):
        """custom skipping message"""
        stdout('Skipping:' , *args , color = 'gray' , **kwargs)

    @classmethod
    def _speical_message(cls , *args , padding_char : None | str = None , padding_width : int = 100 , color : str | None = None , level_prefix : dict[str, Any] | None = None , vb_level : int | None = None , **kwargs):
        """custom cyan colored Highlight level message"""
        assert not level_prefix or not padding_char , 'prefix and padding_char cannot be used together'
        if vb_level is None:
            vb_level = 0 if level_prefix else 1
        msg = ' '.join([str(s) for s in args])
        if padding_char and len(msg) < padding_width:
            padding_left = padding_char * max(0 , (padding_width - len(msg) - 2) // 2)
            padding_right = padding_char * max(0 , padding_width - len(msg) - 2 - len(padding_left))
            msg = ' '.join([padding_left , msg , padding_right])
        if level_prefix:
            stderr(msg , color = color , bold = True , level_prefix = level_prefix , **kwargs)
        else:
            stdout(msg , color = color , bold = True , **kwargs)

    @classmethod
    def remark(cls , *args , color = 'lightblue' , prefix = False , padding_char : None | str = None , padding_width : int = 100 , **kwargs):
        """
        custom blue stdout message for remark
        level: 0 for blue (normal) , 1 for yellow (warning) , 2 for red (error) , 3 for purple (critical)
        """
        level_prefix = {'level' : 'REMARK' , 'color' : 'white' , 'bg_color' : color} if prefix else None
        cls._speical_message(*args , padding_char = padding_char , padding_width = padding_width , color = color , level_prefix = level_prefix , **kwargs)

    @classmethod
    def highlight(cls , *args , color = 'cyan' , prefix = False , padding_char : None | str = None , padding_width : int = 100 , **kwargs):
        """custom cyan colored Highlight level message"""
        level_prefix = {'level' : 'HIGHLIGHT' , 'color' : 'black' , 'bg_color' : color} if prefix else None
        cls._speical_message(*args , padding_char = padding_char , padding_width = padding_width , color = color , level_prefix = level_prefix , **kwargs)

    @classmethod
    def debug(cls , *args , indent : int = 0 , vb_level : int = 0 , **kwargs):
        """Debug level stderr"""
        if vb_level <= ProjConfig.verbosity:
            cls.log.debug(FormatStr(*args , indent = indent) , **kwargs)

    @classmethod
    def info(cls , *args , indent : int = 0 , vb_level : int = 1 , **kwargs):
        """Info level stderr"""
        if vb_level <= ProjConfig.verbosity:
            cls.log.info(FormatStr(*args , indent = indent) , **kwargs)

    @classmethod
    def warning(cls , *args , indent : int = 0 , vb_level : int = 1 , **kwargs):
        """Warning level stderr"""
        if vb_level <= ProjConfig.verbosity:
            cls.log.warning(FormatStr(*args , indent = indent) , **kwargs)

    @classmethod
    def error(cls , *args , indent : int = 0 , vb_level : int = 1 , **kwargs):
        """Error level stderr"""
        if vb_level <= ProjConfig.verbosity:
            cls.log.error(FormatStr(*args , indent = indent) , **kwargs)

    @classmethod
    def critical(cls , *args , indent : int = 0 , vb_level : int = 1 , **kwargs):
        """Critical level stderr"""
        if vb_level <= ProjConfig.verbosity:
            cls.log.critical(FormatStr(*args , indent = indent) , **kwargs)

    @classmethod
    def divider(cls , width : int = 100 , char : Literal['-' , '=' , '*'] = '-' , color : str | None = None , vb_level : int = 0):
        """Divider mesge , use stdout"""
        stdout(char * width , color = color , vb_level = vb_level)

    @classmethod
    def conclude(cls , *args : str , level : _type_levels = 'critical'):
        """Add the message to the conclusions for later use"""
        msg = ' '.join([str(s) for s in args])
        cls._conclusions[level].append(msg)

    @classmethod
    def draw_conclusions(cls):
        """wrap the conclusions: printout , merge into a single string and clear them"""
        conclusion_strs = []
        with cls.ParagraphIII('Final Conclusions'):
            for level , color in zip(cls._levels , cls._levels_palette):
                if not cls._conclusions[level]:
                    continue
                stdout(f'There are {len(cls._conclusions[level])} {level.upper()} Conclusions:' , color = color)
                for conclusion in cls._conclusions[level]:
                    stdout(conclusion , indent = 1)
                    conclusion_strs.append(f'{level.upper()}: {conclusion}')
                cls._conclusions[level].clear()
        return '\n'.join(conclusion_strs)

    @classmethod
    def get_conclusions(cls , type : _type_levels) -> list[str]:
        """Get the conclusions"""
        return cls._conclusions[type]

    class ParagraphI:
        """
        Format Enclosed process to count time
        example:
            with Logger.ParagraphI('Process Name'):
                Logger.info('This is the enclosed process...')
        """
        def __init__(self , title : str , vb_level : int = 0):
            self.title = title
            self.vb_level = vb_level
        def __enter__(self):
            self._init_time = datetime.now()
            self.write(f'{self.title} Start at {self._init_time.strftime("%Y-%m-%d %H:%M:%S")}')
        def __exit__(self, *args): 
            self._end_time = datetime.now()
            self.write(f'{self.title} Finished at {self._end_time.strftime("%Y-%m-%d %H:%M:%S")}! Cost {Duration(self._end_time - self._init_time)}')
        def write(self , message : str):
            Logger.warning(message , vb_level = self.vb_level)

    class ParagraphII:
        """
        Format Enclosed process to count time
        example:
            with Logger.ParagraphII('Process Name'):
                Logger.info('This is the enclosed process...')
        """
        def __init__(self , title : str , vb_level : int = 0):
            self.title = title
            self.vb_level = vb_level
        def __enter__(self):
            self._init_time = datetime.now()
            self.write(f'{self.title} Start at {self._init_time.strftime("%Y-%m-%d %H:%M:%S")}')
        def __exit__(self, *args): 
            self._end_time = datetime.now()
            self.write(f'{self.title} Finished at {self._end_time.strftime("%Y-%m-%d %H:%M:%S")}! Cost {Duration(self._end_time - self._init_time)}')
        def write(self , message : str):
            Logger.info(message , vb_level = self.vb_level)

    class ParagraphIII:
        """
        Format enclosed message
        example:
            with Logger.ParagraphIII('Title'):
                Logger.info('This is the enclosed message...')
        """
        def __init__(self , title : str , width : int = 100 , char : Literal['-' , '=' , '*'] = '*', vb_level : int = 0):
            self.title = title.strip().upper()
            self.width = width
            self.char = char
            self.vb_level = vb_level
        def __enter__(self):
            self._init_time = datetime.now()
            self.write(f'{self.title} Start'.upper())

        def __exit__(self , exc_type , exc_value , traceback):
            self._end_time = datetime.now()
            self.write(f'{self.title} Finished in {Duration(self._end_time - self._init_time)}'.upper())

        def write(self , message : str):
            Logger.highlight(message , padding_char = self.char , padding_width = self.width , vb_level = self.vb_level)

    class Profiler(cProfile.Profile):
        """Profiler class for profiling the code, show the profile result in the best way"""
        def __init__(self, profiling = True , builtins = False , display = True , n_head = 20 , 
                     columns = ['type' , 'name' , 'ncalls', 'cumtime' ,  'tottime' , 'percall' , 'where'] , sort_on = 'cumtime' , highlight = None , output = None ,
                     **kwargs) -> None:
            self.profiling = profiling
            if self.profiling: 
                super().__init__(builtins = builtins) 
            self.display = display
            self.n_head = n_head
            self.columns = columns
            self.sort_on = sort_on
            self.highlight = highlight
            self.output = output

        def __enter__(self):
            if self.profiling: 
                return super().__enter__()
            else:
                return self

        def __exit__(self, type , value , trace):
            if type is not None:
                Logger.error(f'Error in Profiler ' , type , value)
                traceback.print_exc()
            elif self.profiling:
                if self.display:
                    df = self.get_df(sort_on = self.sort_on , highlight = self.highlight , output = self.output)
                    Display(df.loc[:,self.columns].head(self.n_head))
                return super().__exit__(type , value , trace)

        def get_df(self , sort_on = 'cumtime' , highlight = None , output = None):
            """Get the profile result as a pandas dataframe"""
            if not self.profiling: 
                return pd.DataFrame()
            # highlight : 'gp_math_func.py'
            df = pd.DataFrame(
                getattr(self , 'getstats')(), 
                columns=pd.Index(['full_name', 'ncalls', 'ccalls', 'cumtime' ,  'tottime' , 'caller']))
            df['tottime'] = df['tottime'].round(4)
            df['cumtime'] = df['cumtime'].round(4)
            df['full_name'] = df['full_name'].astype(str)
            df_func = pd.DataFrame(
                [self.decompose_func_str(s) for s in df.full_name] , 
                columns = pd.Index(['type' , 'name' , 'where' , 'memory']))
            df = pd.concat([df_func , df],axis=1).sort_values(sort_on,ascending=False)
            df['percall'] = df['cumtime'] / df['ncalls']
            column_order = ['type' , 'name' , 'ncalls', 'ccalls', 'cumtime' ,  'tottime' , 'percall' , 'where' , 'memory' , 'full_name', 'caller']
            df = df.loc[:,column_order]
            if isinstance(highlight , str): 
                df = df[df.full_name.str.find(highlight) > 0]
            if isinstance(output , str): 
                path = PATH.log_profile.joinpath(output).with_suffix('.csv')
                df.to_csv(path)
                Logger.footnote(f'Profile result saved to {path}')
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
