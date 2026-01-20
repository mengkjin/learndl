# please check this path before running the code
import sys , socket , platform , os , torch

from dataclasses import dataclass
from pathlib import Path
from typing import Literal , Any

__all__ = ['MACHINE']

@dataclass
class _Mach:
    name : str
    cuda_server : bool
    main_path : Path
    python_path : str
    mosek_lic_path : Path | None = None
    updatable : bool = False
    emailable : bool = False

    @property
    def belong_to_hfm(self) -> bool:
        return self.name.lower().startswith(('hno' , 'hpo'))
    @property
    def belong_to_jinmeng(self) -> bool:
        return 'jinmeng' in self.name.lower()
    @property
    def hfm_factor_dir(self) -> Path | None:
        return Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha') if self.belong_to_hfm else None

    system_name = platform.system()
    is_linux = system_name == 'Linux' and os.name == 'posix'
    is_windows = system_name == 'Windows'
    is_macos = system_name == 'Darwin'
    

_machine_settings : dict[str , _Mach] = {
    'mengkjin-server' : _Mach(
        name = 'mengkjin-server' , 
        cuda_server = True , 
        main_path = Path('/home/mengkjin/workspace/learndl') , 
        python_path = '/home/mengkjin/workspace/learndl/.venv/bin/python' , 
        mosek_lic_path = Path('/home/mengkjin/mosek/mosek.lic') , 
        updatable = True , 
        emailable = True),
    'HST-jinmeng' : _Mach(
        name = 'HST-jinmeng' , 
        cuda_server = False , 
        main_path = Path('E:/workspace/learndl') , 
        python_path = 'E:/workspace/learndl/.venv/Scripts/python.exe' , 
        mosek_lic_path = Path('C:/Users/Administrator/mosek/mosek.lic') , 
        updatable = True , 
        emailable = True),
    'Mathews-Mac' : _Mach(
        name = 'Mathews-Mac' , 
        cuda_server = False , 
        main_path = Path('/Users/mengkjin/workspace/learndl') , 
        python_path = '/Users/mengkjin/workspace/learndl/.venv/bin/python' , 
        mosek_lic_path = Path('/Users/mengkjin/mosek/mosek.lic') , 
        updatable = False , 
        emailable = False),
}

def _get_best_device():
    """Get the best device for the machine: CUDA, MPS, or CPU"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).upper()
    elif torch.mps.is_available():
        return 'MPS'
    else:
        return 'CPU'

def _get_os_name():
    """Get the OS name"""
    return os.name

class MACHINE:
    """
    Machine setting for the project
    name : str , machine_name
    server : bool , is this machine a server
    main_path : str , main_path of the project
    python_path : str , python_path
    mosek_lic_path : str | None , mosek license path
    updatable : bool , updatable
    emailable : bool , emailable

    belong_to_hfm : bool , belong to HFM
    belong_to_jinmeng : bool , belong to Jinmeng
    hfm_factor_dir : Path | None , HFM factor directory

    platform_server : bool , is this machine a platform server
    platform_reserve : bool , is this machine a platform reserved
    platform_coding : bool , is this machine a platform coding

    """
    name : str = socket.gethostname().split('.')[0]
    system_name = platform.system()

    setting : _Mach = _machine_settings[name]
    assert setting.name == name , f'machine name mismatch: {setting.name} != {name}'
    
    cuda_server = setting.cuda_server
    main_path = setting.main_path
    python_path = setting.python_path
    mosek_lic_path = setting.mosek_lic_path
    updatable = setting.updatable
    emailable = setting.emailable

    belong_to_hfm = setting.belong_to_hfm
    belong_to_jinmeng = setting.belong_to_jinmeng
    hfm_factor_dir = setting.hfm_factor_dir

    is_linux = system_name == 'Linux' and os.name == 'posix'
    is_windows = system_name == 'Windows'
    is_macos = system_name == 'Darwin'
    
    platform_server = name == 'mengkjin-server'
    platform_coding = is_macos

    cpu_count = os.cpu_count() or 1
    max_workers = 40 if platform_server else cpu_count
    best_device = _get_best_device()
    
    assert main_path.exists() , f'main_path not exists: {main_path}'
    assert Path(__file__).is_relative_to(main_path) , f'{__file__} is not in {main_path}'
    sys.path.append(str(main_path))
    assert python_path , f'python_path not set for {name}'

    @classmethod
    def info(cls) -> dict[str, Any]:
        """return the machine info list"""
        return {
            'Machine Name' : cls.name, 
            'Is Server' : cls.cuda_server, 
            'System' : cls.system_name, 
            'Main Path' : cls.main_path, 
            'Python Path' : cls.python_path,
            'Best Device' : cls.best_device,
        }

    @classmethod
    def machine_main_path(cls , machine_name : str) -> Path:
        """Get the main path at another machine"""
        return _machine_settings[machine_name].main_path
    
    @classmethod
    def PATH(cls):
        """Get the PATH of the machine"""
        if not hasattr(cls , '_path'):
            from src.proj.env.path import PATH
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
    def configs(cls , conf_type : Literal['proj' , 'factor' , 'boost' , 'nn' , 'train' , 'trade' , 'reserved' , 'schedule'] , name : str , raise_if_not_exist = True , **kwargs) -> dict:
        """
        Get the configs of the machine
        possible conf_type: proj , factor , boost , nn , train , trade , schedule
        possible suffixes: .json , .yaml , prefer json over yaml
        additional kwargs: e.g. encoding
        """
        PATH = cls.PATH()
        for suffix in ('.json' , '.yaml'):
            if (path := PATH.conf.joinpath(conf_type , f'{name}{suffix}')).exists():
                break
        else:
            if raise_if_not_exist:
                raise FileNotFoundError(f'Config file {conf_type}/{name} does not exist')
            else:
                return {}
        
        additional_kwargs = {}
        #match name:
        #    case 'tushare_indus':
        #        additional_kwargs.update({'encoding' : 'gbk'})
        kwargs = kwargs | additional_kwargs

        if path.suffix == '.yaml':
            return PATH.read_yaml(path , **kwargs)
        elif path.suffix == '.json':
            return PATH.read_json(path , **kwargs)
        else:
            raise ValueError(f'Unsupported config file type: {path.suffix}')