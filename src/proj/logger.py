import colorlog , logging , sys
import logging.handlers
from typing import Any , Generator , Literal , Type

from .path import PATH

_seperator_width = 80
_seperator_char = '*'
_divider_char = '='

class Logger:
    """custom colored log , config at PATH.conf / 'glob' / 'logger.yaml'"""
    _instance : 'Logger | Any' = None
    _cached_messages : dict[Literal['info' , 'warning' , 'error' , 'critical' , 'debug'] , list[str]] = {
        'info' : [] ,
        'warning' : [] ,
        'error' : [] ,
        'critical' : [] ,
        'debug' : [] ,
    }
    def __new__(cls, *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance.init_logger()

    def init_logger(self , test_output = False):
        """Initialize the logger"""
        config_logger = PATH.read_yaml(PATH.conf.joinpath('glob' , 'logger.yaml'))
        new_path = PATH.log_main.joinpath(config_logger['file']['param']['filename'])
        config_logger['file']['param']['filename'] = str(new_path)
        new_path.parent.mkdir(exist_ok=True)
        log = logging.getLogger(config_logger['name'])
        log.setLevel(getattr(logging , config_logger['level']))

        while log.handlers:
            log.handlers[-1].close()
            log.removeHandler(log.handlers[-1])

        for hdname in config_logger['handlers']:
            hdargs = config_logger[hdname]['param']
            hdclass : Type[logging.Handler] = eval(config_logger[hdname]['class'])
            handler = hdclass(**hdargs)
            handler.setLevel(config_logger[hdname]['level'])
            hdformatter = eval(config_logger[hdname]['formatter_class'])(datefmt=config_logger['datefmt'],
                                                                         **config_logger['formatters'][config_logger[hdname]['formatter']])
            handler.setFormatter(hdformatter)
            log.addHandler(handler)

        
        if test_output:
            log.debug('This is the DEBUG    message...')
            log.info('This is the INFO     message...')
            log.warning('This is the WARNING  message...')
            log.error('This is the ERROR    message...')
            log.critical('This is the CRITICAL message...')

        self.log = log
        return self
    
    @classmethod
    def print(cls , *args , **kwargs):
        """Print the message to the stdout"""
        print(*args , **kwargs)

    @classmethod
    def debug(cls , *args , **kwargs):
        """Debug level message"""
        cls().log.debug(*args , **kwargs)
        cls.dump_to_logwriter(*args)
    
    @classmethod
    def info(cls , *args , **kwargs):
        """Info level message"""
        cls().log.info(*args , **kwargs)
        cls.dump_to_logwriter(*args)

    @classmethod
    def warning(cls , *args , **kwargs):
        """Warning level message"""
        cls().log.warning(*args , **kwargs)
        cls.dump_to_logwriter(*args)

    @classmethod
    def error(cls , *args , **kwargs):
        """Error level message"""
        cls().log.error(*args , **kwargs)
        cls.dump_to_logwriter(*args)   

    @classmethod
    def critical(cls , *args , **kwargs):
        """Critical level message"""
        cls().log.critical(*args , **kwargs)
        cls.dump_to_logwriter(*args)

    @classmethod
    def separator(cls , width = _seperator_width , char = _seperator_char):
        """Separator message , use info level"""
        cls().log.info(char * width)

    @classmethod
    def divider(cls , width = _seperator_width , char = _divider_char):
        """Divider message , use info level"""
        cls().log.info(char * width)
        
    @staticmethod
    def dump_to_logwriter(*args):
        """Dump the message to the log writer if sys.stdout has a log attribute"""
        log = getattr(sys.stdout , 'log' , None)
        write = getattr(log , 'write' , None)
        if write:
            write(' '.join([str(s) for s in args]) + '\n')

    @classmethod
    def cache_message(cls , type : Literal['info' , 'warning' , 'error' , 'critical' , 'debug'] , message : str):
        """Add the message to the cache for later use"""
        cls._cached_messages[type].append(message)

    @classmethod
    def iter_cached_messages(cls) -> Generator[tuple[Literal['info' , 'warning' , 'error' , 'critical' , 'debug'] , str] , None , None]:
        """Iterate the cached messages"""
        for type in cls._cached_messages:
            while cls._cached_messages[type]:
                yield type , cls._cached_messages[type].pop(0)

    @classmethod
    def get_cached_messages(cls , type : Literal['info' , 'warning' , 'error' , 'critical' , 'debug']) -> list[str]:
        """Get the cached messages"""
        return cls._cached_messages[type]

    class EnclosedMessage:
        """F
        ormat Enclosed message
        example:
            with Logger.EnclosedMessage('Title'):
                Logger.info('This is the enclosed message...')
        """
        def __init__(self , title : str , width = _seperator_width):
            self.title = title.strip()
            self.width = width

        def __enter__(self):
            self.write(f'{self.title.upper()} START')

        def __exit__(self , exc_type , exc_value , traceback):
            self.write(f'{self.title.upper()} FINISH')

        def __repr__(self):
            return f'EnclosedMessage(title = {self.title} , width = {self.width})'
        
        def write(self , message : str):
            txt_len = len(message)
            if txt_len >= self.width:
                Logger.info(message)
            else:
                padding_left = '*' * max(0 , (self.width - txt_len - 2) // 2)
                padding_right = '*' * max(0 , self.width - txt_len - 2 - len(padding_left))
                Logger.info(' '.join([padding_left , message , padding_right]))


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