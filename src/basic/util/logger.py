import logging , sys
import colorlog

from src.basic import path as PATH
from src.basic import conf as CONF

class Logger(logging.RootLogger):
    '''custom colored log , config at {PATH.conf}/logger.yaml '''
    _instance = None
    def __new__(cls, *args , **kwargs):
        if cls._instance is None:
            cls._instance = cls._init_logger()
        return cls._instance

    @staticmethod
    def _init_logger(test_output = False):
        config_logger = CONF.glob('logger')
        new_path = PATH.log_main.joinpath(config_logger['file']['param']['filename'])
        config_logger['file']['param']['filename'] = str(new_path)
        new_path.parent.mkdir(exist_ok=True)
        log = logging.getLogger(config_logger['name'])
        exec("log.setLevel(logging."+config_logger['level']+")")

        while log.handlers:
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
        return log
    
class _LevelFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, level_fmts={}):
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
    def __init__(self, fmt=None, datefmt=None, log_colors={},level_fmts={},secondary_log_colors={}):
        self._level_formatters = {}
        for level, format in level_fmts.items():
            # Could optionally support level names too
            self._level_formatters[getattr(logging , level)] = colorlog.ColoredFormatter(fmt=format, datefmt=datefmt , log_colors=log_colors , secondary_log_colors=secondary_log_colors)
        # self._fmt will be the default format
        super(_LevelColorFormatter, self).__init__(fmt=fmt, datefmt=datefmt,log_colors=log_colors,secondary_log_colors=secondary_log_colors)

    def format(self, record):
        if record.levelno in self._level_formatters:
            return self._level_formatters[record.levelno].format(record)
        return super(_LevelColorFormatter, self).format(record)

class DualPrinter:
    '''change print target to both terminal and file'''
    def __init__(self, filename : str):
        self.filename = PATH.log_update.joinpath(filename)
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
