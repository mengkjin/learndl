import torch
from pathlib import Path
from ..project_setting import MACHINE

from . import path as PATH
from . import conf as CONF
from .util import *

for name in dir(PATH):
    member = getattr(PATH , name)
    if isinstance(member , Path) and member.is_relative_to(PATH.main):
        member.mkdir(parents=True , exist_ok=True)

print(f'main path: {MACHINE.project_path}')
if torch.cuda.is_available():
    print(f'Use device name: ' + torch.cuda.get_device_name(0))
elif MACHINE.server:
    print('server should have cuda , please check the cuda status')
