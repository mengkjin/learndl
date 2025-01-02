# please check this path before running the code
import sys , socket , torch , yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class MachineSetting:
    server : bool
    path: str

    def initialize(self):
        self.name = socket.gethostname()
        assert self.path , f'main_path not set for {self.name}'
        assert Path(self.path).exists() , f'MAIN_PATH not exists: {self.path}'
        assert (not self.server) or torch.cuda.is_available() , f'server should have cuda'
        assert Path(__file__).is_relative_to(Path(self.path)) , f'{__file__} is not in {self.path}'
        sys.path.append(self.path)
        print(f'main path: {self.path}')
        if torch.cuda.is_available(): print(f'Use device name: ' + torch.cuda.get_device_name(0))
        return self
    
    @classmethod
    def MachineDict(cls):
        return {
            'mengkjin-server':  cls(True , '/home/mengkjin/workspace/learndl'),
            'longcl-server':    cls(True , '/home/longcl/workspace/learndl'),
            'zhuhy-server':     cls(True , '/home/zhuhy/workspace/learndl'),
            'HNO-JINMENG01':    cls(False , 'D:/Coding/learndl/learndl'),
            'HPO-LONGCL05':     cls(False , ''),
            'HPO-ZHUHY01':      cls(False , ''),
            'HST-jinmeng':      cls(False , 'E:/workspace/learndl'),
        }
    
    @classmethod
    def select_machine(cls):
        machine_name = socket.gethostname()
        machine_dict = cls.MachineDict()
        assert machine_name in machine_dict , f'unidentified machine: {machine_name} , please check the MachineDict method'
        return machine_dict[machine_name].initialize()
    
    @property
    def belong_to_hfm(self):
        return self.name.lower().startswith(('hno' , 'hpo'))
    
    @property
    def belong_to_jinmeng(self): # perform rcode transfer
        return 'jinmeng' in self.name.lower()
    
    @property
    def hfm_factor_dir(self): # perform rcode transfer
        return Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha') if self.belong_to_hfm else None

MACHINE = MachineSetting.select_machine()