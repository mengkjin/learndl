# please check this path before running the code
import sys , socket

from pathlib import Path
from typing import Literal

_machine_dict = {
    # machine name :    (is_server , main_path , updateable)
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

class MACHINE:
    """
    Machine setting for the project
    name : str , machine_name
    server : bool , is this machine a server
    main_path : str , main_path of the project
    updateable : bool , updateable
    python_path : str , python_path

    belong_to_hfm : bool , belong to HFM
    belong_to_jinmeng : bool , belong to Jinmeng
    hfm_factor_dir : Path | None , HFM factor directory
    """
    name : str = socket.gethostname().split('.')[0]
    settings : tuple = _machine_dict[name]
    server : bool = settings[0]
    main_path : Path = Path(settings[1])
    updateable : bool = settings[2] if len(settings) > 2 else True
    python_path : str = _get_python_path(name , main_path)

    belong_to_hfm : bool = name.lower().startswith(('hno' , 'hpo'))
    belong_to_jinmeng : bool = 'jinmeng' in name.lower()
    hfm_factor_dir : Path | None = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha') if belong_to_hfm else None
    
    assert main_path.exists() , f'main_path not exists: {main_path}'
    assert Path(__file__).is_relative_to(main_path) , f'{__file__} is not in {main_path}'
    sys.path.append(str(main_path))
    assert python_path , f'python_path not set for {name}'

    @classmethod
    def select_machine(cls):
        """Select the machine setting"""
        return cls
    
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
        PATH = cls.PATH()
        p = PATH.local_settings.joinpath(f'{name}.yaml')
        if not p.exists():
            raise FileNotFoundError(f'{p} does not exist , .local_settings folder only has {[p.stem for p in PATH.list_files(PATH.local_settings)]}')
        return PATH.read_yaml(p)

    @classmethod
    def get_share_folder_path(cls) -> Path | None:
        """Get the share folder path of the machine"""
        try:
            return Path(cls.local_settings('share_folder')[cls.name])
        except (FileNotFoundError , KeyError):
            return None
        
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
