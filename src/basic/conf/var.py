import torch

from .. import path as PATH
from ..global_env import GLOB_ENV

def glob(name : str):
    if name == 'tushare_indus': kwargs = {'encoding' : GLOB_ENV['tushare_indus_encoding']}
    else: kwargs = {}
    p = PATH.conf.joinpath('glob' , f'{name}.yaml')
    assert p.exists() , p
    return PATH.read_yaml(p , **kwargs)

SILENT = GLOB_ENV['silent']

THIS_IS_SERVER : bool = torch.cuda.is_available() # socket.gethostname() == 'mengkjin-server'
assert not THIS_IS_SERVER or torch.cuda.is_available() , f'SERVER must have cuda available'