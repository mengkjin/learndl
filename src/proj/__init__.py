from .env import MACHINE , PATH , ProjStates , Options
from .func import Timer , PTimer , Duration , Display , Silence
from .util import (
    Logger , IOCatcher , LogWriter , OutputCatcher , OutputDeflector , 
    HtmlCatcher , MarkdownCatcher , WarningCatcher , DBConnHandler , 
    SharedSync , Email , Device , MemoryPrinter)

def print_project_info():
    import torch , os
    identifier = 'project_initialized'
    if identifier in os.environ:
        return
    
    Logger.highlight(f'Project Initialized Successfully!')
    [Logger.success(info) for info in MACHINE.info() + ProjStates.info()]
    if MACHINE.server and not torch.cuda.is_available():
        Logger.error(f'[{MACHINE.name}] server should have cuda but not available, please check the cuda status')
    os.environ[identifier] = "1"

print_project_info()