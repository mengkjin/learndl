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

class ConfFileLazyLoader:
    """Lazy loader for config files"""
    _root_path : Path
    def __init__(self, name: str , root_path: Path):
        self._name = name
        self._root_path = root_path
        self._contents : dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, main_key: str , sub_key: str | None = None , **kwargs) -> Any:
        return self.get(main_key, sub_key, **kwargs)

    def get(self, main_key: str , sub_key: str | None = None , **kwargs) -> Any:
        """
        Get the content of the config file by the main key and sub key
        will lazy load the content of main_key if not loaded yet
        """
        content = self._get_content(main_key)
        if not sub_key:
            return content
        for arg in sub_key.split('/'):
            if isinstance(content, dict):
                if arg not in content and 'default' in kwargs:
                    return kwargs['default']
                content = content[arg]
            else:
                raise ValueError(f'{self.name} {content} is not a dict , cannot get {arg} from sub_key {sub_key}')
        return content

    def _get_content(self, key: str) -> dict:
        """Load all ``.yaml`` / ``.json`` files under ``root_path`` into a stem-keyed dict.

        Returns:
            Mapping of filename stem to parsed content (dict/list/scalar per file).

        Raises:
            ValueError: If a non-file entry exists under ``root_path``.
            AssertionError: If a file is not YAML or JSON.
        """
        if key in self._contents:
            return self._contents[key]

        for suffix in ('.json' , '.yaml'):
            if (path := self._root_path.joinpath(*key.split('/')).with_suffix(suffix)).exists():
                break
        else:
            raise FileNotFoundError(f'{self.name} file {key} does not exist')
        
        if path.suffix == '.yaml':
            with open(path , 'r') as f:
                content = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path , 'r') as f:
                content = json.load(f)

        self._contents[key] = content
        return content

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
    secret = ConfFileLazyLoader('Secret' , MAIN_PATH.joinpath('.secret'))
    config = ConfFileLazyLoader('Config' , MAIN_PATH.joinpath('configs'))

    setting = _MachineSettings(**secret.get('machines' , name))
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
        return Path(cls.secret.get('machines' , f'{machine_name}/main_path'))