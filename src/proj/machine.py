# please check this path before running the code
import sys , socket

from pathlib import Path
from typing import Literal

class MachineSetting:
    MACHINE_DICT = {
        # machine name :    (is_server , project_path , updateable)
        'mengkjin-server':  (True , '/home/mengkjin/workspace/learndl'),
        'longcl-server':    (True , '/home/longcl/workspace/learndl'),
        'zhuhy-server':     (True , '/home/zhuhy/workspace/learndl'),
        'HNO-JINMENG01':    (False , 'D:/Coding/learndl/learndl'),
        'HPO-LONGCL05':     (False , ''),
        'HPO-ZHUHY01':      (False , ''),
        'HST-jinmeng':      (False , 'E:/workspace/learndl'),
        'Mathews-Mac':      (False , '/Users/mengkjin/workspace/learndl' , False),
    }

    def __init__(self , server : bool , project_path : str , updateable : bool = True):
        self.server = server
        self.project_path = project_path
        self.updateable = updateable
        self.initialize()

    def initialize(self):
        assert self.project_path , f'main_path not set for {self.name}'
        assert Path(self.project_path).exists() , f'MAIN_PATH not exists: {self.project_path}'
        assert Path(__file__).is_relative_to(Path(self.project_path)) , f'{__file__} is not in {self.project_path}'
        sys.path.append(self.project_path)
        return self
    
    @property
    def name(self):
        return self.get_machine_name()
    
    @classmethod
    def select_machine(cls):
        machine_name = cls.get_machine_name()
        assert machine_name in cls.MACHINE_DICT , f'unidentified machine: {machine_name} , please check the MACHINE_DICT attribute'
        machine = cls(*cls.MACHINE_DICT[machine_name])
        return machine
    
    @classmethod
    def get_main_path(cls):
        return Path(cls.MACHINE_DICT[cls.get_machine_name()][1])
    
    @classmethod
    def get_machine_name(cls):
        return socket.gethostname().split('.')[0]
    
    @property
    def belong_to_hfm(self):
        return self.name.lower().startswith(('hno' , 'hpo'))
    
    @property
    def belong_to_jinmeng(self): # perform rcode transfer
        return 'jinmeng' in self.name.lower()
    
    @property
    def hfm_factor_dir(self): # perform rcode transfer
        return Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha') if self.belong_to_hfm else None
    
    @property
    def python_path(self):
        if self.name in ['Mathews-Mac']:
            return self.project_path + '/.venv/bin/python'
        elif self.name in ['HST-jinmeng']:
            return 'E:/workspace/learndl/.venv/Scripts/python.exe'
        elif self.name in ['mengkjin-server']:
            return self.project_path + '/.venv/bin/python' #'python3.10'
        else:
            return 'python'
        
    @property
    def PATH(self):
        if not hasattr(self , '_PATH'):
            from src.proj import PATH
            self._PATH = PATH
        return self._PATH

    def local_settings(self , name : str):
        p = self.PATH.local_settings.joinpath(f'{name}.yaml')
        assert p.exists() , f'{p} does not exist , .local_settings folder only has {[p.stem for p in self.PATH.list_files(self.PATH.local_settings)]}'
        return self.PATH.read_yaml(p)
    
    @property
    def share_folder_path(self):
        try:
            return Path(self.local_settings('share_folder')[self.name])
        except:
            return None
        
    def configs(self , conf_type : Literal['glob' , 'registry' , 'factor' , 'boost' , 'nn' , 'train' , 'trade' , 'schedule'] , name : str):
        p = self.PATH.conf.joinpath(conf_type , f'{name}.yaml')
        assert p.exists() , p
        if conf_type == 'glob' and name == 'tushare_indus': 
            kwargs = {'encoding' : 'gbk'}
        else: 
            kwargs = {}
        return self.PATH.read_yaml(p , **kwargs)

MACHINE = MachineSetting.select_machine()