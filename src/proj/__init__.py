from .machine import MACHINE
from .path import PATH
from .logger import Logger
from .output_catcher import IOCatcher , LogWriter , OutputCatcher , OutputDeflector
from .message_catcher import HtmlCatcher , MarkdownCatcher
from .db import DBConnHandler
from .warning_catcher import WarningCatcher
from .instance_record import InstanceRecord
from .shared_sync import SharedSync

def print_project_info():
    import torch , os
    identifier = 'project_initialized'
    if identifier in os.environ:
        return

    print(f'main path: {MACHINE.main_path}')
    if torch.cuda.is_available():
        print(f'Use device name: ' + torch.cuda.get_device_name(0))
    elif MACHINE.server:
        print('server should have cuda , please check the cuda status')
    elif torch.mps.is_available():
        print('Use MPS as default device')
    else:
        print('Use CPU as default device')

    print(f'src.proj.InstanceRecord can be accessed to check {InstanceRecord._slots}')
    os.environ[identifier] = "1"

print_project_info()