from .env import MACHINE , PATH , ProjStates , ProjConfig
from .func import Timer , PTimer , Duration , Display , Silence , FormatStr , stdout , stderr
from .util import (
    Logger , IOCatcher , LogWriter , 
    HtmlCatcher , MarkdownCatcher , WarningCatcher , DBConnHandler , 
    SharedSync , Email , Device , MemoryPrinter , Options)

def print_project_info(script_level : bool = False , identifier = 'project_initialized'):
    import torch , os
    if identifier in os.environ and not script_level:
        return
    elif getattr(ProjStates , identifier , False) and script_level:
        return
    
    Logger.highlight(f'Project Initialized Successfully!')
    [Logger.success(info) for info in MACHINE.info() + ProjConfig.info() + ProjStates.info()]
    if MACHINE.server and not torch.cuda.is_available():
        Logger.error(f'[{MACHINE.name}] server should have cuda but not available, please check the cuda status')
    os.environ[identifier] = "1"
    setattr(ProjStates , identifier , True)

print_project_info()