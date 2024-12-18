from pathlib import Path
from src.project_setting import *

from . import path as PATH
from . import conf as CONF
from .util import *

for name in dir(PATH):
    member = getattr(PATH , name)
    if isinstance(member , Path) and member.is_relative_to(PATH.main):
        member.mkdir(parents=True , exist_ok=True)


# print some info after import basic
print('Basic module imported!')
if torch.cuda.is_available(): print(f'Use device name: ' + torch.cuda.get_device_name(0))
