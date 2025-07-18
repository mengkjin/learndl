import colorlog , logging , sys
import logging.handlers
from typing import Any , Type
from pathlib import Path

from src.basic import path as PATH
from src.basic import conf as CONF

class Logger:
    '''custom colored log (Only one instance) , config at {PATH.conf}/logger.yaml '''
    _instance : 'Logger | Any' = None
    def __new__(cls, *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance.init_logger()

    def init_logger(self , test_output = False):
        config_logger = CONF.glob('logger')
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
    def separator(cls , width = 80 , char = '-'):
        cls().log.info(char * width)
        
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
            Logger.separator(self.width)
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


class LogWriter:
    class TeeOutput:
        """double output stream: output to console and file"""
        def __init__(self, original_stream, log):
            self.original = original_stream
            self.log = log

        def __repr__(self):
            return f'original: {self.original} log: {self.log}'
            
        def write(self, message):
            self.original.write(message)
            self.original.flush()
            self.log.write(message)
            
        def flush(self):
            self.original.flush()
    

    '''change print target to both terminal and file'''
    def __init__(self, filename : str | Path | None = None):
        self.set_attrs(filename)

    def set_attrs(self , filename : str | Path | None = None):
        if isinstance(filename , str): filename = Path(filename)
        self.filename = filename
        if self.filename is None: return
        self.filename.parent.mkdir(exist_ok=True,parents=True)
        self.log = open(self.filename, "w")
        return self

    def __enter__(self):
        assert self.filename is not None , 'filename is not set'
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.TeeOutput(self.original_stdout , self.log)
        sys.stderr = self.TeeOutput(self.original_stderr , self.log)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log.close()

    def contents(self):
        assert self.filename is not None , 'filename is not set'
        with open(self.filename , 'r') as f:
            return f.read()