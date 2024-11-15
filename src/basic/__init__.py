import torch
from . import conf as CONF
from . import path as PATH

from . import util
from .util import SILENT , CALENDAR , TradeDate , Timer

from .path import THIS_IS_SERVER
assert not THIS_IS_SERVER or torch.cuda.is_available() , f'SERVER must have cuda available'