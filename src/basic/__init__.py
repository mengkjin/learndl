import torch
from . import conf as CONF
from . import path as PATH
from . import util
from .util import SILENT

# variables
THIS_IS_SERVER  = torch.cuda.is_available() # socket.gethostname() == 'mengkjin-server'
# assertions
assert not THIS_IS_SERVER or torch.cuda.is_available() , f'SERVER must have cuda available'