import torch
from pathlib import Path
from ..proj import MACHINE as machine

from . import db as DB
from . import conf as CONF
from .util import *

print(f'main path: {machine.project_path}')
if torch.cuda.is_available():
    print(f'Use device name: ' + torch.cuda.get_device_name(0))
elif machine.server:
    print('server should have cuda , please check the cuda status')
