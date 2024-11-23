from pathlib import Path
from .project_setting import *

from . import path as PATH
[p.mkdir(parents=True , exist_ok=True) for nm in dir(PATH) if isinstance(p:= getattr(PATH , nm) , Path) and p.is_relative_to(PATH.main)]

from . import conf as CONF
from .util import *