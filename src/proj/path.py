# please check this path before running the code
import shutil , yaml , time , os
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
    scpt        = main.joinpath('src' , 'scripts')
    fac_def     = main.joinpath('src' , 'res' , 'defs' , 'factor')
    pool_def    = main.joinpath('src' , 'res' , 'defs' , 'pooling')

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
    log_update  = logs.joinpath('update')
    log_autorun = logs.joinpath('autorun')

    # models folder and subfolders
    model       = main.joinpath('models')

    # configs folder and subfolders
    conf        = main.joinpath('configs')
    conf_schedule = conf.joinpath('schedule')

    # results folder and subfolders
    result      = main.joinpath('results')
    rslt_train  = result.joinpath('train')
    rslt_factor = result.joinpath('test').joinpath('perf')
    rslt_optim  = result.joinpath('test').joinpath('optim')
    rslt_top    = result.joinpath('test').joinpath('top')
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
    def read_yaml(yaml_file : str | Path , **kwargs) -> Any:
        """Read yaml file"""
        if isinstance(yaml_file , str):
            yaml_file = Path(yaml_file)
        if isinstance(yaml_file , Path) and yaml_file.suffix == '' and yaml_file.with_name(f'{yaml_file.name}.yaml').exists():
            yaml_file = yaml_file.with_name(f'{yaml_file.name}.yaml')
        if not yaml_file.exists():
            if yaml_file.parent.stem == 'nn':
                print(f'{yaml_file} does not exist, trying default.yaml')
                yaml_file = yaml_file.with_name(f'default.yaml')
        with open(yaml_file ,'r' , **kwargs) as f:
            d = yaml.load(f , Loader = yaml.FullLoader)
        return d

    @staticmethod
    def dump_yaml(data , yaml_file , **kwargs) -> None:
        """Dump data to yaml file"""
        with open(yaml_file , 'a' if os.path.exists(yaml_file) else 'w') as f:
            yaml.dump(data , f , **kwargs)

    @staticmethod
    def copytree(src : str | Path , dst : str | Path) -> None:
        """Copy entire directory"""
        shutil.copytree(src , dst)

    @staticmethod
    def copyfiles(src : str | Path , dst : str | Path , bases : list[str]) -> None:
        """Copy files from source to destination"""
        [shutil.copyfile(f'{src}/{base}' , f'{dst}/{base}') for base in bases]

    @staticmethod
    def deltrees(dir : str | Path , bases : list[str] , verbose = True) -> None:
        """Delete sub folders in the directory"""
        for base in bases:
            if verbose: 
                print(f'Deleting {base} in {dir}')
            shutil.rmtree(f'{dir}/{base}')

    @staticmethod
    def file_modified_date(path : Path | str , default = 19970101) -> int:
        """Get the modified date of the file"""
        if Path(path).exists():
            return int(time.strftime('%Y%m%d',time.localtime(os.path.getmtime(path))))
        else:
            return default

    @staticmethod
    def file_modified_time(path : Path | str , default = 19970101000000) -> int:
        """Get the modified time of the file"""
        if Path(path).exists():
            return int(time.strftime('%Y%m%d%H%M%S',time.localtime(os.path.getmtime(path))))
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