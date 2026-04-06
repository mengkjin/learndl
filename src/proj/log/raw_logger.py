"""Build project loggers from machine ``setting/logger`` config (handlers, formatters)."""

import colorlog , logging
import logging.handlers

from typing import Any , Type

from src.proj import PATH , CONST

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
    """Load logger config and rewrite file path under ``PATH.logs/main``.

    Returns:
        Config dict suitable for ``new_log`` / ``reset_logger``.
    """
    log_config = CONST.Pref.load_preference('logger')
    new_path = PATH.logs.joinpath('main' , log_config['file']['param']['filename'])
    log_config['file']['param']['filename'] = str(new_path)
    new_path.parent.mkdir(exist_ok=True)

    return log_config

def new_log(config : dict[str, Any]) -> logging.Logger:
    """Create (or reconfigure) a named logger from ``config``."""
    log = logging.getLogger(config['name'])
    log.setLevel(getattr(logging , config['level']))
    reset_logger(log , config)
    return log

def reset_logger(log : logging.Logger , config : dict[str, Any]):
    """Replace all handlers on ``log`` using ``config`` handlers and formatters.

    Returns:
        The same ``log`` instance after reconfiguration.
    """
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
