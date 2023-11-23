import os
import logging
import colorlog
from logging import handlers
import yaml

'''
LOG_CONFIG = {
    'name' : 'default_log',
    'handlers': ['console' , 'file'],
    'level': 'DEBUG',
    'datefmt' : '%y-%m-%d %H:%M:%S',
    # 处理器集合
    'console': {
        'level': 'INFO',  # 输出信息的最低级别
        'class': 'logging.StreamHandler',
        'param' : {},
        'formatter_class' : '_LevelColorFormatter', 
        # 'colorlog.ColoredFormatter' , '_LevelFormatter' , 'logging.Formatter'
        'formatter': 'levelcolor',  # 'color' , 'level' , 'standard'
    },
    # 输出到文件
    'file': {
        'level': 'DEBUG',
        'class': 'logging.handlers.TimedRotatingFileHandler',
        'param' : {
            'filename' : './logs/nn_fac_log.log',
            'when' : 'D',
            'backupCount': 5,  # 备份份数
            'encoding': 'utf-8',  # 文件编码
        },
        'formatter_class' : '_LevelFormatter',
        'formatter': 'level', 
    },
    # 日志格式集合
    'formatters': {
        # 标准输出格式 , omit part : 'TRD:%(threadName)-10s|LVL:%(levelno)s|'
        'standard': {
            'fmt': '%(asctime)s|MOD:%(module)-12s|: %(message)s',
        },
        'level' : {
            'fmt': '%(asctime)s|MOD:%(module)-12s|: %(message)s',
            'level_fmts' : {
                'DEBUG' : '%(message)s',
                'INFO' : '%(message)s',
            },
        },
        'color' : {
            'fmt': '%(log_color)s%(asctime)s|MOD:%(module)-12s|%(reset_log_color)s: %(message_log_color)s%(message)s',
            'log_colors' : {
                'DEBUG':'bold,white,bg_cyan',
                'INFO':'bold,white,bg_green',
                'WARNING':'bold,white,bg_blue',
                'ERROR':'bold,white,bg_purple',
                'CRITICAL':'bold,white,bg_red',
            },
            'secondary_log_colors' : {
                'reset': {
                    'DEBUG':'reset',
                    'INFO':'reset',
                    'WARNING':'reset',
                    'ERROR':'reset',
                    'CRITICAL':'reset',
                },
                'message': {
                    'DEBUG':'cyan',
                    'INFO':'green',
                    'WARNING':'bold,blue',
                    'ERROR':'bold,purple',
                    'CRITICAL':'bold,red',
                },
            },
        },
        'levelcolor' : {
            #'fmt': '%(log_color)s%(asctime)s|MOD:%(module)-12s|TRD:%(threadName)-12s|LVL:%(levelno)s|%(reset_log_color)s: %(message_log_color)s%(message)s',
            'fmt': '%(log_color)s%(asctime)s|MOD:%(module)-12s|%(reset_log_color)s: %(message_log_color)s%(message)s',
            'level_fmts' : {
                'DEBUG' : '%(message_log_color)s%(message)s',
                'INFO' : '%(message_log_color)s%(message)s',
            },
            'log_colors' : {
                'DEBUG':'bold,white,bg_cyan',
                'INFO':'bold,white,bg_green',
                'WARNING':'bold,white,bg_blue',
                'ERROR':'bold,white,bg_purple',
                'CRITICAL':'bold,white,bg_red',
            },
            'secondary_log_colors' : {
                'reset': {
                    'DEBUG':'reset',
                    'INFO':'reset',
                    'WARNING':'reset',
                    'ERROR':'reset',
                    'CRITICAL':'reset',
                },
                'message': {
                    'DEBUG':'cyan',
                    'INFO':'green',
                    'WARNING':'bold,blue',
                    'ERROR':'bold,purple',
                    'CRITICAL':'bold,red',
                },
            },
        },
    },
}
'''

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

def get_config(config_files = ['data_type' , 'train']):
    config_dict = dict()
    if isinstance(config_files , str): config_files = [config_files]
    for cfg_name in config_files:
        with open(f'./configs/config_{cfg_name}.yaml' ,'r') as f:
            cfg = yaml.load(f , Loader = yaml.FullLoader)
        if cfg_name == 'train':
            if 'SPECIAL_CONFIG' in cfg.keys() and 'SHORTTEST' in cfg['SPECIAL_CONFIG'].keys(): 
                if cfg['SHORTTEST']: cfg.update(cfg['SPECIAL_CONFIG']['SHORTTEST'])
                del cfg['SPECIAL_CONFIG']['SHORTTEST']
            if 'SPECIAL_CONFIG' in cfg.keys() and 'TRANSFORMER' in cfg['SPECIAL_CONFIG'].keys():
                if cfg['MODEL_MODULE'] == 'Transformer' or (cfg['MODEL_MODULE'] in ['GeneralRNN'] and 'transformer' in cfg['MODEL_PARAM']['type_rnn']):
                    cfg['TRAIN_PARAM']['trainer'].update(cfg['SPECIAL_CONFIG']['TRANSFORMER']['trainer'])
                del cfg['SPECIAL_CONFIG']['TRANSFORMER']
        config_dict.update(cfg)
    return config_dict