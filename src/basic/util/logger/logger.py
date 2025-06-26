import colorlog , logging , sys

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
        self.dual_printer(*args)
    
    def info(self , *args , **kwargs):
        self.log.info(*args , **kwargs)
        self.dual_printer(*args)

    def warning(self , *args , **kwargs):
        self.log.warning(*args , **kwargs)
        self.dual_printer(*args)

    def error(self , *args , **kwargs):
        self.log.error(*args , **kwargs)
        self.dual_printer(*args)   

    def critical(self , *args , **kwargs):
        self.log.critical(*args , **kwargs)
        self.dual_printer(*args)
        
    def dual_printer(self , *args):
        log = getattr(sys.stdout , 'log' , None)
        write = getattr(log , 'write' , None)
        if write:
            write(' '.join([str(s) for s in args]) + '\n')


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
        def __init__(self, original_stream, log_file):
            self.original = original_stream
            self.log_file = log_file
            
        def write(self, message):
            self.original.write(message)
            self.original.flush()
            self.log_file.write(message)
            
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