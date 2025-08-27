from pathlib import Path
from ..project_setting import *

from . import path as PATH
from . import conf as CONF
from .util import *

for name in dir(PATH):
    member = getattr(PATH , name)
    if isinstance(member , Path) and member.is_relative_to(PATH.main):
        member.mkdir(parents=True , exist_ok=True)
