# please check this path before running the code
import shutil , yaml , os , sys , json

from datetime import datetime
from pathlib import Path
from typing import Any

from .machine import MACHINE

def _initialize_path(obj : object) -> None:
    """Initialize the all paths under the main path"""
    for name in dir(obj):
        member = getattr(obj , name)
        if isinstance(member , Path) and member.is_relative_to(MACHINE.main_path):
            member.mkdir(parents=True , exist_ok=True)

class PATH:
    """
    Path structure for the project
    examples:
        PATH.main
        PATH.scpt
        PATH.fac_def
        PATH.conf
    """
    main        = MACHINE.main_path
    scpt        = main.joinpath('scripts')
    fac_def     = main.joinpath('src' , 'res' , 'factor' , 'defs')

    # data folder and subfolders
    data        = main.joinpath('data')
    database    = data.joinpath('DataBase')
    export      = data.joinpath('Export')
    interim     = data.joinpath('Interim')
    miscel      = data.joinpath('Miscellaneous')
    updater     = data.joinpath('Updater')

    block       = interim.joinpath('DataBlock')
    batch       = interim.joinpath('MiniBatch')
    datacache   = interim.joinpath('DataCache')
    norm        = interim.joinpath('HistNorm')

    hidden      = export.joinpath('hidden_feature')
    factor      = export.joinpath('stock_factor')
    pred        = export.joinpath('model_prediction')
    fmp         = export.joinpath('factor_model_port')
    fmp_account = export.joinpath('factor_model_account')
    trade_port  = export.joinpath('trading_portfolio')

    # logs folder and subfolders
    logs        = main.joinpath('logs')
    log_main    = logs.joinpath('main')
    log_catcher = logs.joinpath('catcher')
    log_profile = logs.joinpath('profile')

    # models folder and subfolders
    model       = main.joinpath('models')
    
    # configs folder and subfolders
    conf        = main.joinpath('configs')

    # results folder and subfolders
    result      = main.joinpath('results')
    null_model  = result.joinpath('null_models')
    rslt_train  = result.joinpath('train')
    rslt_test   = result.joinpath('test')
    rslt_trade  = result.joinpath('trade')

    # resouces folder (for update)
    resource   = main.joinpath('resources')
    bak_data   = resource.joinpath('tushare_bak_data')
    bak_record = resource.joinpath('tushare_bak_data_record')

    # local_resources folder
    local_resources = main.joinpath('.local_resources')
    temp            = local_resources.joinpath('temp')
    local_machine   = local_resources.joinpath(MACHINE.name)
    local_shared    = local_resources.joinpath('shared')
    shared_schedule = local_shared.joinpath('schedule_model')
    app_db          = local_machine.joinpath('app_db')
    runtime         = local_machine.joinpath('runtime')
    optuna          = local_machine.joinpath('optuna')
    tensorboard     = local_machine.joinpath('tensorboard')

    # local_settings folder
    local_settings = main.joinpath('.local_settings')

    @classmethod
    def path_at_machine(cls , path : Path | str , machine_name : str) -> str | Path:
        """Get the path at another machine"""
        if isinstance(path , str):
            path = Path(path)
            return str(cls.path_at_machine(Path(path) , machine_name))
        else:
            if path.is_relative_to(MACHINE.main_path):
                return MACHINE.machine_main_path(machine_name).joinpath(path.relative_to(MACHINE.main_path))
            else:
                return path
        
    @staticmethod
    def read_yaml(yaml_file : str | Path , **kwargs) -> dict[str, Any]:
        """Read yaml file"""
        yaml_file = Path(yaml_file)
        if yaml_file.suffix == '' and yaml_file.with_suffix('.yaml').exists():
            yaml_file = yaml_file.with_suffix('.yaml')
        if not yaml_file.exists():
            sys.stderr.write(f'\u001b[31m\u001b[1m{yaml_file} does not exist!\u001b[0m\n')
            return {}
        with open(yaml_file ,'r' , **kwargs) as f:
            d = yaml.load(f , Loader = yaml.FullLoader)
        return d

    @staticmethod
    def dump_yaml(data , yaml_file : str | Path , **kwargs) -> None:
        """Dump data to yaml file"""
        assert isinstance(data , dict) , type(data)
        yaml_file = Path(yaml_file)
        assert yaml_file.suffix == '.yaml' , yaml_file
        assert not yaml_file.exists() or not os.path.getsize(yaml_file) , f'{yaml_file} already exists'
        with open(yaml_file , 'a' if os.path.exists(yaml_file) else 'w') as f:
            yaml.dump(data , f , **kwargs)

    @staticmethod
    def read_json(json_file : str | Path , **kwargs) -> dict[str, Any]:
        """Read json file"""
        json_file = Path(json_file)
        if json_file.suffix == '' and json_file.with_suffix('.json').exists():
            json_file = json_file.with_suffix('.json')
        if not json_file.exists():
            sys.stderr.write(f'\u001b[31m\u001b[1m{json_file} does not exist!\u001b[0m\n')
            return {}
        with open(json_file , 'r' , **kwargs) as f:
            d = json.load(f)
        return d

    @staticmethod
    def dump_json(data , json_file : str | Path , ensure_ascii = False , indent = 4 , **kwargs) -> None:
        """Dump data to json file"""
        assert isinstance(data , dict) , type(data)
        json_file = Path(json_file)
        assert json_file.suffix == '.json' , json_file
        assert not json_file.exists() or not os.path.getsize(json_file) , f'{json_file} already exists'
        with open(json_file , 'w' , **kwargs) as f:
            json.dump(data , f , ensure_ascii = ensure_ascii , indent = indent , **kwargs)

    @staticmethod
    def copytree(src : str | Path , dst : str | Path) -> None:
        """Copy entire directory"""
        shutil.copytree(src , dst)

    @staticmethod
    def copyfiles(src : str | Path , dst : str | Path , bases : list[str]) -> None:
        """Copy files from source to destination"""
        [shutil.copyfile(f'{src}/{base}' , f'{dst}/{base}') for base in bases]

    @staticmethod
    def deltrees(dir : str | Path , bases : list[str]) -> None:
        """Delete sub folders in the directory"""
        for base in bases:
            sys.stderr.write(f'\u001b[31m\u001b[1mDeleting {base} in {dir}\u001b[0m\n')
            shutil.rmtree(f'{dir}/{base}')

    @staticmethod
    def file_modified_date(path : Path | str , default = 19970101) -> int:
        """Get the modified date of the file"""
        if Path(path).exists():
            return int(datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y%m%d'))
        else:
            return default

    @staticmethod
    def file_modified_time(path : Path | str , default = 19970101000000) -> int:
        """Get the modified time of the file"""
        if Path(path).exists():
            return int(datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y%m%d%H%M%S'))
        else:
            return default
        
    @classmethod
    def initialize_path(cls) -> None:
        """Initialize the all paths under the main path"""
        for name in dir(cls):
            member = getattr(cls , name)
            if isinstance(member , Path) and member.is_relative_to(cls.main):
                member.mkdir(parents=True , exist_ok=True)

    @classmethod
    def list_files(cls , directory : str | Path , fullname = False , recur = False) -> list[Path]:
        """List all files in directory"""
        if isinstance(directory , str): 
            directory = Path(directory)
        if recur:
            paths : list[Path] = []
            paths = [Path(dirpath).joinpath(filename) for dirpath, _, filenames in os.walk(directory) for filename in filenames]
        else:
            paths = [p for p in directory.iterdir()]
        paths = [p.absolute() for p in paths] if fullname else [p.relative_to(directory) for p in paths]
        paths = cls.filter_paths(paths)
        return paths

    @staticmethod
    def filter_paths(paths : list[Path] , ignore_prefix = ('.' , '~')) -> list[Path]:
        """Filter paths by ignoring certain prefixes"""
        return [p for p in paths if not p.name.startswith(ignore_prefix)]

    @classmethod
    def get_local_settings(cls , name : str) -> dict:
        """Get the local settings of the machine"""
        p = cls.local_settings.joinpath(f'{name}.yaml')
        if not p.exists():
            raise FileNotFoundError(f'{p} does not exist , .local_settings folder only has {[p.stem for p in cls.list_files(cls.local_settings)]}')
        return cls.read_yaml(p)

    @classmethod
    def get_share_folder_path(cls) -> Path | None:
        """Get the share folder path of the machine"""
        try:
            return Path(cls.get_local_settings('share_folder')[MACHINE.name])
        except (FileNotFoundError , KeyError):
            return None
            
_initialize_path(PATH)