# please check this path before running the code
import shutil , yaml , sys , json

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
    log_model   = logs.joinpath('model')

    # results folder and subfolders
    result      = main.joinpath('results')
    rslt_factor = result.joinpath('factor')
    rslt_trade  = result.joinpath('trade')

    # models folder and subfolders
    model       = main.joinpath('models')
    model_nn    = model.joinpath('nn')
    model_boost = model.joinpath('boost')
    model_factor= model.joinpath('factor')
    model_st    = model.joinpath('st')
    
    # configs folder and subfolders
    conf        = main.joinpath('configs')

    # resouces folder (for update)
    resource   = main.joinpath('resources')
    bak_data   = resource.joinpath('tushare_bak_data')
    bak_record = resource.joinpath('tushare_bak_data_record')

    # local_resources folder
    local_resources = main.joinpath('.local_resources')
    temp            = local_resources.joinpath('temp')
    local_machine   = local_resources.joinpath(MACHINE.name)
    app_db          = local_machine.joinpath('app_db')
    runtime         = local_machine.joinpath('runtime')
    optuna          = local_machine.joinpath('optuna')
    tensorboard     = local_machine.joinpath('tensorboard')

    share_folder       = MACHINE.share_folder
    local_share_folder = local_resources.joinpath('shared')
    shared_schedule    = local_share_folder.joinpath('schedule_model')

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
            d = yaml.safe_load(f)
        return d

    class IndentedDumper(yaml.Dumper):
        """add indent for yaml list"""
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    @classmethod
    def dump_yaml(cls , data , yaml_file : str | Path , * , indent = 2 , overwrite = False , **kwargs) -> None:
        """Dump data to yaml file"""
        assert isinstance(data , dict) , type(data)
        yaml_file = Path(yaml_file)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        assert yaml_file.suffix == '.yaml' , yaml_file
        assert not yaml_file.exists() or not yaml_file.stat().st_size or overwrite , f'{yaml_file} already exists'
        with open(yaml_file , 'w') as f:
            yaml.dump(data ,  f , Dumper = cls.IndentedDumper , indent = indent , **kwargs)

    @staticmethod
    def read_json(json_file : str | Path , encoding = 'utf-8' , **kwargs) -> dict[str, Any]:
        """Read json file"""
        json_file = Path(json_file)
        if json_file.suffix == '' and json_file.with_suffix('.json').exists():
            json_file = json_file.with_suffix('.json')
        if not json_file.exists():
            sys.stderr.write(f'\u001b[31m\u001b[1m{json_file} does not exist!\u001b[0m\n')
            return {}
        with open(json_file , 'r' , encoding = encoding , **kwargs) as f:
            d = json.load(f)
        return d

    @staticmethod
    def dump_json(data , json_file : str | Path , ensure_ascii = False , indent = 4 , overwrite = False , **kwargs) -> None:
        """Dump data to json file"""
        assert isinstance(data , dict) , type(data)
        json_file = Path(json_file)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        assert json_file.suffix == '.json' , json_file
        assert not json_file.exists() or not json_file.stat().st_size or overwrite , f'{json_file} already exists'
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
        """Get the modified date of the file in '%Y%m%d' format"""
        path = Path(path)
        if path.exists():
            return int(datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y%m%d'))
        else:
            return default

    @staticmethod
    def file_modified_time(path : Path | str , default = 19970101000000) -> int:
        """Get the modified time of the file in '%Y%m%d%H%M%S' format"""
        path = Path(path)
        if path.exists():
            return int(datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y%m%d%H%M%S'))
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
            paths = [p for p in directory.rglob('*') if p.is_file()]
        else:
            paths = [p for p in directory.iterdir()]
        paths = [p.absolute() for p in paths] if fullname else [p.relative_to(directory) for p in paths]
        paths = cls.filter_paths(paths)
        return paths

    @staticmethod
    def filter_paths(paths : list[Path] , ignore_prefix = ('.' , '~')) -> list[Path]:
        """Filter paths by ignoring certain prefixes"""
        return [p for p in paths if not p.name.startswith(ignore_prefix)]
            
_initialize_path(PATH)