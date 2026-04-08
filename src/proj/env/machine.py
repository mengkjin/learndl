"""Machine identity, OS, secrets, and config loading for the current host."""

import sys , socket , platform , os , torch , pytz , yaml , json

from dataclasses import dataclass
from pathlib import Path
from typing import Any , Literal
from tzlocal import get_localzone

__all__ = ['MACHINE']

def get_project_root() -> Path:
    """Get the project root path of the project, depending on the pyproject.toml file"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("pyproject.toml not found, please confirm the project root directory contains this file")

MAIN_PATH = get_project_root()
SECRET_PATH = MAIN_PATH.joinpath('.secret')
@dataclass
class _MachineSettings:
    name : str
    cuda_server : bool
    main_path : str
    python_path : str
    share_folder : str | None = None
    mosek_lic_path : str | None = None
    updatable : bool = False
    emailable : bool = False
    nickname : str = ''

    def __post_init__(self):
        if not self.nickname:
            self.nickname = self.name
        assert Path(self.main_path) == MAIN_PATH , f'main_path {self.main_path} is not the same as {MAIN_PATH}'

    @property
    def belong_to_hfm(self) -> bool:
        return self.name.lower().startswith(('hno' , 'hpo'))
    @property
    def belong_to_jinmeng(self) -> bool:
        return 'jinmeng' in self.name.lower()
    @property
    def hfm_factor_dir(self) -> Path | None:
        return Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha') if self.belong_to_hfm else None


def _get_system_name():
    """Normalize OS name to ``linux`` | ``windows`` | ``macos``.

    Returns:
        Short platform tag used for ``MACHINE.system_name``.

    Raises:
        ValueError: If the platform is not supported.
    """
    system_name = platform.system()
    if system_name == 'Linux' and os.name == 'posix':
        return 'linux'
    elif system_name == 'Windows':
        return 'windows'
    elif system_name == 'Darwin':
        return 'macos'
    else:
        raise ValueError(f'Unsupported system name: {system_name}')

def _get_best_device():
    """Get the best device for the machine: CUDA, MPS, or CPU"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).upper()
    elif torch.mps.is_available():
        return 'MPS'
    else:
        return 'CPU'

def _get_secret() -> dict:
    """Load all ``.yaml`` / ``.json`` files under ``.secret`` into a stem-keyed dict.

    Returns:
        Mapping of filename stem to parsed content (dict/list/scalar per file).

    Raises:
        ValueError: If a non-file entry exists under ``SECRET_PATH``.
        AssertionError: If a file is not YAML or JSON.
    """
    secret = {}
    for file in SECRET_PATH.iterdir():
        if file.is_file():
            assert file.suffix == '.yaml' or file.suffix == '.json' , f'{file} is not a yaml or json file'
            if file.suffix == '.yaml':
                with open(file , 'r') as f:
                    secret[file.stem] = yaml.safe_load(f)
            elif file.suffix == '.json':
                with open(file , 'r') as f:
                    secret[file.stem] = json.load(f)
        else:
            raise ValueError(f'{file} is not a file')
    return secret

class MACHINE:
    """
    Machine setting for the project
    name : str , machine_name
    system_name : str , system name
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
    system_name : Literal['linux' , 'windows' , 'macos'] = _get_system_name()
    
    main_path = MAIN_PATH
    secret = _get_secret()

    setting = _MachineSettings(**secret['machines'][name])
    assert setting.name == name , f'machine name mismatch: {setting.name} != {name}'
    
    cuda_server = setting.cuda_server
    python_path = setting.python_path
    share_folder = Path(setting.share_folder) if setting.share_folder else None
    mosek_lic_path = Path(setting.mosek_lic_path) if setting.mosek_lic_path else None
    updatable = setting.updatable
    emailable = setting.emailable
    nickname = setting.nickname

    belong_to_hfm = setting.belong_to_hfm
    belong_to_jinmeng = setting.belong_to_jinmeng
    hfm_factor_dir = setting.hfm_factor_dir

    is_linux = system_name == 'linux'
    is_windows = system_name == 'windows'
    is_macos = system_name == 'macos'
    
    platform_server = name == 'mengkjin-server'
    platform_coding = is_macos

    cpu_count = os.cpu_count() or 1
    max_workers = 40 if platform_server else cpu_count
    best_device = _get_best_device()

    timezone = get_localzone()
    utc8 = str(timezone) == str(pytz.timezone('Asia/Shanghai') )
    
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
            'Timezone' : cls.timezone,
            'Main Path' : cls.main_path, 
            'Python Path' : cls.python_path,
            'Best Device' : cls.best_device,
        }

    @classmethod
    def machine_main_path(cls , machine_name : str) -> Path:
        """Get the main path at another machine"""
        return Path(cls.secret['machines'][machine_name]['main_path'])
    
    @classmethod
    def PATH(cls):
        """Get the PATH of the machine"""
        if not hasattr(cls , '_path'):
            from src.proj.env.path import PATH
            cls._path = PATH
        return cls._path
        
    @classmethod
    def configs(cls , *args , raise_if_not_exist = True , **kwargs) -> dict:
        """
        Get the configs of the machine
        possible suffixes: .json , .yaml , prefer json over yaml
        additional kwargs: e.g. encoding
        """
        PATH = cls.PATH()
        path = PATH.conf.joinpath(*args)
        if path.is_dir():
            raise TypeError(f'Config {"/".join(args)} is a directory')
        for suffix in ('.json' , '.yaml'):
            if (path := PATH.conf.joinpath(*args).with_suffix(suffix)).exists():
                break
        else:
            if raise_if_not_exist:
                raise FileNotFoundError(f'Config file {"/".join(args)} does not exist')
            else:
                return {}
        
        additional_kwargs = {}
        kwargs = kwargs | additional_kwargs

        if path.suffix == '.yaml':
            return PATH.read_yaml(path , **kwargs)
        elif path.suffix == '.json':
            return PATH.read_json(path , **kwargs)
        else:
            raise ValueError(f'Unsupported config file type: {path.suffix}')