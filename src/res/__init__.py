'''import all modules , order matters'''
from . import algo , deap
from . import factor
from . import model
from . import trading

from . import api

def _print_init_info():
    from src.proj import MACHINE
    import torch
    print(f'main path: {MACHINE.project_path}')
    if torch.cuda.is_available():
        print(f'Use device name: ' + torch.cuda.get_device_name(0))
    elif MACHINE.server:
        print('server should have cuda , please check the cuda status')

_print_init_info()