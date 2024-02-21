import os , shutil
import logging
from logging import handlers
from typing import Any
import colorlog
import yaml

_current_dir = os.path.dirname(os.path.abspath(__file__)) 
DIR_main = f'{_current_dir}/../..'
DIR_data = f'{_current_dir}/../../data'
DIR_conf = f'{_current_dir}/../../configs'
DIR_logs = f'{_current_dir}/../../logs'

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

def get_logger(test_output = False):
    config_logger = get_config('logger')
    config_logger['file']['param']['filename'] = '/'.join([DIR_logs,config_logger['file']['param']['filename']])
    os.makedirs(os.path.dirname(config_logger['file']['param']['filename']), exist_ok = True)
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

def get_config(config_files = ['train']):
    config_dict = dict()
    if isinstance(config_files , str): config_files = [config_files]
   
    for cfg_name in config_files:
        cfg_file = cfg_name if cfg_name.endswith('.yaml') else f'{DIR_conf}/config_{cfg_name}.yaml'
        with open(cfg_file ,'r') as f:
            cfg = yaml.load(f , Loader = yaml.FullLoader)
        config_dict.update(cfg)
    return config_dict
