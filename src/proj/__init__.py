from .machine import MACHINE
from .path import PATH
from .logger import Logger
from .output_catcher import IOCatcher , LogWriter , OutputCatcher , OutputDeflector
from .message_catcher import HtmlCatcher , MarkdownCatcher
from .db import DBConnHandler

def _print_init_info():
    import torch
    print(f'main path: {MACHINE.project_path}')
    if torch.cuda.is_available():
        print(f'Use device name: ' + torch.cuda.get_device_name(0))
    elif MACHINE.server:
        print('server should have cuda , please check the cuda status')
    elif torch.mps.is_available():
        print('Use MPS as default device')
    else:
        print('Use CPU as default device')

_print_init_info()