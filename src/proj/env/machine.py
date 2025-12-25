# please check this path before running the code
import sys , socket , platform , os , torch

from pathlib import Path
from typing import Literal

__all__ = ['MACHINE']

_machine_dict = {
    # machine name :    (is_server , main_path , updatable)
    'mengkjin-server':  (True , '/home/mengkjin/workspace/learndl'),
    'HST-jinmeng':      (False , 'E:/workspace/learndl'),
    'Mathews-Mac':      (False , '/Users/mengkjin/workspace/learndl' , False),
    'longcl-server':    (True , '/home/longcl/workspace/learndl'),
    'zhuhy-server':     (True , '/home/zhuhy/workspace/learndl'),
    'HNO-JINMENG01':    (False , 'D:/Coding/learndl/learndl'),
    'HPO-LONGCL05':     (False , ''),
    'HPO-ZHUHY01':      (False , ''),
}

def _get_python_path(machine_name : str , main_path : Path):
    """Get the Python path of the machine"""
    if machine_name in ['Mathews-Mac']:
        return str(main_path) + '/.venv/bin/python'
    elif machine_name in ['HST-jinmeng']:
        return 'E:/workspace/learndl/.venv/Scripts/python.exe'
    elif machine_name in ['mengkjin-server']:
        return str(main_path) + '/.venv/bin/python' #'python3.10'
    else:
        return 'python'

def _get_best_device():
    """Get the best device for the machine: CUDA, MPS, or CPU"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).upper()
    elif torch.mps.is_available():
        return 'MPS'
    else:
        return 'CPU'

class MACHINE:
    """
    Machine setting for the project
    name : str , machine_name
    server : bool , is this machine a server
    main_path : str , main_path of the project
    updatable : bool , updatable
    python_path : str , python_path

    belong_to_hfm : bool , belong to HFM
    belong_to_jinmeng : bool , belong to Jinmeng
    hfm_factor_dir : Path | None , HFM factor directory
    """
    name : str = socket.gethostname().split('.')[0]
    settings : tuple = _machine_dict[name]
    server : bool = settings[0]
    main_path : Path = Path(settings[1])
    updatable : bool = settings[2] if len(settings) > 2 else True
    python_path : str = _get_python_path(name , main_path)

    belong_to_hfm : bool = name.lower().startswith(('hno' , 'hpo'))
    belong_to_jinmeng : bool = 'jinmeng' in name.lower()
    hfm_factor_dir : Path | None = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha') if belong_to_hfm else None

    system_name = platform.system()
    is_linux = system_name == 'Linux' and os.name == 'posix'
    is_windows = system_name == 'Windows'
    is_macos = system_name == 'Darwin'

    best_device = _get_best_device()
    
    assert main_path.exists() , f'main_path not exists: {main_path}'
    assert Path(__file__).is_relative_to(main_path) , f'{__file__} is not in {main_path}'
    sys.path.append(str(main_path))
    assert python_path , f'python_path not set for {name}'

    @classmethod
    def info(cls) -> list[str]:
        """return the machine info list"""
        return [
            f'Machine Name   : {cls.name}', 
            f'Is Server      : {cls.server}', 
            f'System         : {cls.system_name}', 
            f'Main Path      : {cls.main_path}' , 
            f'Python Path    : {cls.python_path}' ,
            f'Best Device    : {cls.best_device}' ,
        ]

    @classmethod
    def machine_names(cls):
        """Select the machine setting"""
        return list(_machine_dict.keys())

    @classmethod
    def machine_main_path(cls , machine_name : str) -> Path:
        """Get the main path at another machine"""
        return Path(_machine_dict[machine_name][1])
    
    @classmethod
    def PATH(cls):
        """Get the PATH of the machine"""
        if not hasattr(cls , '_path'):
            from src.proj import PATH
            cls._path = PATH
        return cls._path

    @classmethod
    def local_settings(cls , name : str) -> dict:
        """Get the local settings of the machine"""
        return cls.PATH().get_local_settings(name)

    @classmethod
    def share_folder_path(cls) -> Path | None:
        """Get the share folder path of the machine"""
        return cls.PATH().get_share_folder_path()
        
    @classmethod
    def configs(cls , conf_type : Literal['glob' , 'registry' , 'factor' , 'boost' , 'nn' , 'train' , 'trade' , 'schedule'] , name : str) -> dict:
        """Get the configs of the machine"""
        PATH = cls.PATH()
        p = PATH.conf.joinpath(conf_type , f'{name}.yaml')
        assert p.exists() , p
        if conf_type == 'glob' and name == 'tushare_indus': 
            kwargs = {'encoding' : 'gbk'}
        else: 
            kwargs = {}
        return PATH.read_yaml(p , **kwargs)