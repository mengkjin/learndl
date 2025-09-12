# please check this path before running the code
import shutil , yaml , time , os
from pathlib import Path

from .machine import MachineSetting

class PathStructure:
    main        = Path(MachineSetting.get_main_path())
    scpt        = main.joinpath('src' , 'scripts')
    fac_def     = main.joinpath('src' , 'res' , 'facdef')
    conf        = main.joinpath('configs')

    # data folder and subfolders
    data        = main.joinpath('data')
    database    = data.joinpath('DataBase')
    export      = data.joinpath('Export')
    interim     = data.joinpath('Interim')
    miscel      = data.joinpath('Miscellaneous')
    updater     = data.joinpath('Updater')

    block       = interim.joinpath('DataBlock')
    batch       = interim.joinpath('MiniBatch')
    dataset     = interim.joinpath('DataSet')
    norm        = interim.joinpath('HistNorm')

    hidden      = export.joinpath('hidden_feature')
    factor      = export.joinpath('stock_factor')
    preds       = export.joinpath('model_prediction')
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
    local_machine   = local_resources.joinpath(MachineSetting.get_machine_name())
    app_db          = local_machine.joinpath('app_db')
    runtime         = local_machine.joinpath('runtime')
    optuna          = local_machine.joinpath('optuna')
    tensorboard     = local_machine.joinpath('tensorboard')

    # local_settings folder
    local_settings = main.joinpath('.local_settings')

    def __init__(self):
        self.initialize_path()

    @staticmethod
    def read_yaml(yaml_file : str | Path , **kwargs):
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
    def dump_yaml(data , yaml_file , **kwargs):
        with open(yaml_file , 'a' if os.path.exists(yaml_file) else 'w') as f:
            yaml.dump(data , f , **kwargs)

    @staticmethod
    def copytree(src , dst):
        shutil.copytree(src , dst)

    @staticmethod
    def copyfiles(src , dst , bases):
        [shutil.copyfile(f'{src}/{base}' , f'{dst}/{base}') for base in bases]

    @staticmethod
    def deltrees(dir , bases , verbose = True):
        for base in bases:
            if verbose: 
                print(f'Deleting {base} in {dir}')
            shutil.rmtree(f'{dir}/{base}')

    @staticmethod
    def file_modified_date(path : Path | str , default = 19970101):
        if Path(path).exists():
            return int(time.strftime('%Y%m%d',time.localtime(os.path.getmtime(path))))
        else:
            return default

    @staticmethod
    def file_modified_time(path : Path | str , default = 19970101000000):
        if Path(path).exists():
            return int(time.strftime('%Y%m%d%H%M%S',time.localtime(os.path.getmtime(path))))
        else:
            return default
        
    @classmethod
    def initialize_path(cls):
        for name in dir(cls):
            member = getattr(cls , name)
            if isinstance(member , Path) and member.is_relative_to(cls.main):
                member.mkdir(parents=True , exist_ok=True)

    @classmethod
    def list_files(cls , directory : str | Path , fullname = False , recur = False):
        '''list all files in directory'''
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
    def filter_paths(paths : list[Path] , ignore_prefix = ('.' , '~')):
        return [p for p in paths if not p.name.startswith(ignore_prefix)]
            
PATH = PathStructure()
