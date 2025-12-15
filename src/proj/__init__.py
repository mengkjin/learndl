from .machine import MACHINE
from .path import PATH
from .logger import Logger
from .catcher import (
    IOCatcher , LogWriter , OutputCatcher , OutputDeflector , 
    HtmlCatcher , MarkdownCatcher , WarningCatcher)
from .sqlite import DBConnHandler
from .debug import InstanceRecord
from .shared_sync import SharedSync
from .silence import SILENT
from .timer import Duration , Timer , PTimer
from .options import Options
from .display import Display

def print_project_info():
    import torch , os
    identifier = 'project_initialized'
    if identifier in os.environ:
        return

    Logger.highlight(f'Project Initialized Successfully!')
    [Logger.success(info) for info in MACHINE.machine_info()]
    if MACHINE.server and not torch.cuda.is_available():
        Logger.error(f'[{MACHINE.name}] server should have cuda but not available, please check the cuda status')

    # Logger.debug(f'src.proj.InstanceRecord can be accessed to check {InstanceRecord._slots}')
    os.environ[identifier] = "1"

print_project_info()