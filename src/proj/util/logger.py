import colorlog , logging , sys
import logging.handlers

import re
import cProfile
import traceback

import pandas as pd

from datetime import datetime
from typing import Any , Generator , Literal , Type

from src.proj.env import PATH
from src.proj.func import Duration , Display

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
    """custom colored log , config at PATH.conf / 'glob' / 'logger.yaml'"""
    _instance : 'Logger | Any' = None
    _type_levels = Literal['info' , 'debug' , 'warning' , 'error' , 'critical']
    _levels : list[_type_levels] = ['info' , 'debug' , 'warning' , 'error' , 'critical']
    _cached_messages : dict[_type_levels , list[str]] = {level : [] for level in _levels}
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
    def mark(cls , *args , **kwargs):
        """dump to log writer with no display"""
        cls.dump_to_logwriter(*args)

    @classmethod
    def stdout(cls , *args , **kwargs):
        """custom stdout message"""
        print(*args , **kwargs)

    @classmethod
    def success(cls , *args , **kwargs):
        """custom green colored Mark level message"""
        sys.stdout.write(f"\u001b[32m{' '.join(args)}\u001b[0m" + '\n')

    @classmethod
    def fail(cls , *args , **kwargs):
        """custom red colored Fail level message"""
        sys.stderr.write(f"\u001b[31m{' '.join(args)}\u001b[0m" + '\n')

    @classmethod
    def warn(cls , *args , **kwargs):
        """custom yellow colored Warning level message"""
        sys.stderr.write(f"\u001b[33m{' '.join(args)}\u001b[0m" + '\n')

    @classmethod
    def highlight(cls , *args , default_prefix = False , **kwargs):
        """custom cyan colored Highlight level message"""
        msg = ' '.join(args)
        msg = f'\u001b[36m\u001b[1m{msg}\u001b[0m'
        if default_prefix:
            prefix = f'{datetime.now().strftime("%y-%m-%d %H:%M:%S")}|LEVEL:{"HIGHLIGHT":9s}|'
            prefix = f'\u001b[46m\u001b[1m\u001b[30m{prefix}\u001b[0m'
            msg = f'{prefix}: {msg}'
        sys.stderr.write(msg + '\n')

    @classmethod
    def debug(cls , *args , **kwargs):
        """Debug level message"""
        cls.log.debug(*args , **kwargs)
        cls.dump_to_logwriter(*args)
    
    @classmethod
    def info(cls , *args , **kwargs):
        """Info level message"""
        cls.log.info(*args , **kwargs)
        cls.dump_to_logwriter(*args)

    @classmethod
    def warning(cls , *args , **kwargs):
        """Warning level message"""
        cls.log.warning(*args , **kwargs)
        cls.dump_to_logwriter(*args)

    @classmethod
    def error(cls , *args , **kwargs):
        """Error level message"""
        cls.log.error(*args , **kwargs)
        cls.dump_to_logwriter(*args)   

    @classmethod
    def critical(cls , *args , **kwargs):
        """Critical level message"""
        cls.log.critical(*args , **kwargs)
        cls.dump_to_logwriter(*args)

    @classmethod
    def divider(cls , width : int = 100 , char : Literal['-' , '=' , '*'] = '-'):
        """Divider message , use info level"""
        print(char * width)
        
    @staticmethod
    def dump_to_logwriter(*args):
        """Dump the message to the log writer if sys.stdout has a log attribute"""
        log = getattr(sys.stdout , 'log' , None)
        write = getattr(log , 'write' , None)
        if write:
            write(' '.join([str(s) for s in args]) + '\n')

    @classmethod
    def cache_message(cls , type : _type_levels , message : str):
        """Add the message to the cache for later use"""
        cls._cached_messages[type].append(message)

    @classmethod
    def iter_cached_messages(cls) -> Generator[tuple[_type_levels , str] , None , None]:
        """Iterate the cached messages"""
        for type in cls._cached_messages:
            while cls._cached_messages[type]:
                yield type , cls._cached_messages[type].pop(0)

    @classmethod
    def get_cached_messages(cls , type : _type_levels) -> list[str]:
        """Get the cached messages"""
        return cls._cached_messages[type]

    class ParagraphI:
        """
        Format Enclosed process to count time
        example:
            with Logger.ParagraphI('Process Name'):
                Logger.info('This is the enclosed process...')
        """
        def __init__(self , title : str):
            self.title = title
        def __enter__(self):
            self._init_time = datetime.now()
            self.write(f'{self.title} Start at {self._init_time.strftime("%Y-%m-%d %H:%M:%S")}')
        def __exit__(self, *args): 
            self._end_time = datetime.now()
            self.write(f'{self.title} Finished at {self._end_time.strftime("%Y-%m-%d %H:%M:%S")}! Cost {Duration(self._end_time - self._init_time)}')
        def write(self , message : str):
            Logger.critical(message)

    class ParagraphII:
        """
        Format Enclosed process to count time
        example:
            with Logger.ParagraphII('Process Name'):
                Logger.info('This is the enclosed process...')
        """
        def __init__(self , title : str):
            self.title = title
        def __enter__(self):
            self._init_time = datetime.now()
            self.write(f'{self.title} Start at {self._init_time.strftime("%Y-%m-%d %H:%M:%S")}')
        def __exit__(self, *args): 
            self._end_time = datetime.now()
            self.write(f'{self.title} Finished at {self._end_time.strftime("%Y-%m-%d %H:%M:%S")}! Cost {Duration(self._end_time - self._init_time)}')
        def write(self , message : str):
            Logger.warning(message)

    class ParagraphIII:
        """
        Format enclosed message
        example:
            with Logger.ParagraphIII('Title'):
                Logger.info('This is the enclosed message...')
        """
        def __init__(self , title : str , width : int = 100 , char : Literal['-' , '=' , '*'] = '*'):
            self.title = title.strip().upper()
            self.width = width
            self.char = char
        def __enter__(self):
            self._init_time = datetime.now()
            self.write(f'{self.title} Start'.upper())

        def __exit__(self , exc_type , exc_value , traceback):
            self._end_time = datetime.now()
            self.write(f'{self.title} Finished in {Duration(self._end_time - self._init_time)}'.upper())

        def write(self , message : str):
            txt_len = len(message)
            if txt_len >= self.width:
                Logger.highlight(message)
            else:
                padding_left = self.char * max(0 , (self.width - txt_len - 2) // 2)
                padding_right = self.char * max(0 , self.width - txt_len - 2 - len(padding_left))
                Logger.highlight(' '.join([padding_left , message , padding_right]))

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
                print(f'Error in Profiler ' , type , value)
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
            df.tottime = df.tottime.round(4)
            df.cumtime = df.cumtime.round(4)
            df.full_name = df.full_name.astype(str)
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
                print(f'Profile result saved to {path}')
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
                    #    print(func_string)
                    break
            if data is None: 
                print(func_string)
                data = [''] * 4
            data[2] = data[2].replace(str(PATH.main) , '')
            return data
