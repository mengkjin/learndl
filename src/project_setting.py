# please check this path before running the code
import sys , socket
from pathlib import Path

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
        if self.name in ['Mathews-Mac' , 'HST-jinmeng']:
            return self.project_path + '/.venv/bin/python'
        elif self.name in ['mengkjin-server']:
            return 'python3.10'
        else:
            return 'python'

MACHINE = MachineSetting.select_machine()