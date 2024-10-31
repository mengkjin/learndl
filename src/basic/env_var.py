import torch
from pathlib import Path

class Silence:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self): 
        self._silent : bool = False
        self._enable : bool = True
    @property
    def silent(self): 
        return self._silent and self._enable
    def __bool__(self): 
        return self.silent
    def __enter__(self) -> None: 
        self._raw_silent = self._silent
        self._silent = True
    def __exit__(self , *args) -> None: 
        self._silent = self._raw_silent
    def disable(self): 
        self._enable = False
    def enable(self): 
        self._enable = True

# variables
MAIN_PATH       = [parent for parent in list(Path(__file__).parents) if parent.match('./src/')][-1].parent
SILENT          = Silence()
THIS_IS_SERVER  = torch.cuda.is_available() # socket.gethostname() == 'mengkjin-server'

FACTOR_DESTINATION_LAPTOP = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha')
FACTOR_DESTINATION_SERVER = MAIN_PATH.joinpath('results' , 'Alpha')

# assertions
assert not THIS_IS_SERVER or torch.cuda.is_available() , f'SERVER must have cuda available'