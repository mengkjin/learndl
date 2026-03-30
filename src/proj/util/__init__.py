"""Shared utilities: HTTP, SQLite, plotting, scripts, proxies, email, and catchers."""

from src.proj.abc.singleton import SingletonABCMeta

from .basic import *
from .func import *

from .catcher import (
    IOCatcher , LogWriter , WarningCatcher ,
    HtmlCatcher , MarkdownCatcher , CrashProtectorCatcher)
from .shared_sync import SharedSync
from .email import Email
from .device import Device , MemoryPrinter
from .options import Options
from .sqlite import DBConnHandler
from .http import http_session

from .script import *
from .proxy import ProxyAPI

from . import plot as Plot

