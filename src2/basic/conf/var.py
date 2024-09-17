import torch
from .. import path as PATH

def glob(name : str):
    if name == 'tushare_indus': kwargs = {'encoding' : 'gbk'}
    else: kwargs = {}
    p = PATH.conf.joinpath('glob' , f'{name}.yaml')
    assert p.exists() , p
    return PATH.read_yaml(p , **kwargs)

class _Silence:
    def __init__(self) -> None:
        self.silent = False

    def __bool__(self): return self.silent

    def __enter__(self) -> None: 
        self.silent = True
    def __exit__(self , *args) -> None: 
        self.silent = False

SILENT = _Silence()

THIS_IS_SERVER : bool = torch.cuda.is_available() # socket.gethostname() == 'mengkjin-server'
assert not THIS_IS_SERVER or torch.cuda.is_available() , f'SERVER must have cuda available'