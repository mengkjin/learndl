import colorlog , logging , sys
import logging.handlers
from typing import Any , Type

from .path import PATH

class Logger:
    '''custom colored log (Only one instance) , config at {PATH.conf}/logger.yaml '''
    _instance : 'Logger | Any' = None
    def __new__(cls, *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance.init_logger()

    def init_logger(self , test_output = False):
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
        print(*args , **kwargs)

    @classmethod
    def debug(cls , *args , **kwargs):
        cls().log.debug(*args , **kwargs)
        cls.dump_to_logwriter(*args)
    
    @classmethod
    def info(cls , *args , **kwargs):
        cls().log.info(*args , **kwargs)
        cls.dump_to_logwriter(*args)

    @classmethod
    def warning(cls , *args , **kwargs):
        cls().log.warning(*args , **kwargs)
        cls.dump_to_logwriter(*args)

    @classmethod
    def error(cls , *args , **kwargs):
        cls().log.error(*args , **kwargs)
        cls.dump_to_logwriter(*args)   

    @classmethod
    def critical(cls , *args , **kwargs):
        cls().log.critical(*args , **kwargs)
        cls.dump_to_logwriter(*args)

    @classmethod
    def separator(cls , width = 80 , char = '*'):
        cls().log.info(char * width)

    @classmethod
    def divider(cls):
        cls().separator(80 , '=')
        
    @staticmethod
    def dump_to_logwriter(*args):
        log = getattr(sys.stdout , 'log' , None)
        write = getattr(log , 'write' , None)
        if write:
            write(' '.join([str(s) for s in args]) + '\n')

    class EnclosedMessage:
        def __init__(self , title : str , width = 80):
            self.title = title
            self.width = width

        def __enter__(self):
            if len(self.title) >= self.width:
                Logger.info(self.title.upper())
            else:
                padding = '*' * ((self.width - len(self.title)) // 2)
                Logger.info(padding + self.title.upper() + padding)

        def __exit__(self , exc_type , exc_value , traceback):
            Logger.separator(self.width)


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