import os , shutil , yaml , time
from pathlib import Path

from src.project_setting import MACHINE

# variables
main        = Path(MACHINE.path)

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

# config folder and subfolders
conf        = main.joinpath('configs')

# logs folder and subfolders
logs        = main.joinpath('logs')
log_main    = logs.joinpath('main')
log_optuna  = logs.joinpath('optuna')
log_update  = logs.joinpath('update')
log_tensor  = logs.joinpath('tensorboard')

# models folder and subfolders
model       = main.joinpath('models')

# results folder and subfolders
result      = main.joinpath('results')
rslt_factor = result.joinpath('test').joinpath('perf')
rslt_optim  = result.joinpath('test').joinpath('optim')
rslt_top    = result.joinpath('test').joinpath('top')

monitor     = result.joinpath('monitor')

# some records to upload
upload      = main.joinpath('upload')

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

def dump_yaml(data , yaml_file , **kwargs):
    with open(yaml_file , 'a' if os.path.exists(yaml_file) else 'w') as f:
        yaml.dump(data , f , **kwargs)

def copytree(src , dst):
    shutil.copytree(src , dst)

def copyfiles(src , dst , bases):
    [shutil.copyfile(f'{src}/{base}' , f'{dst}/{base}') for base in bases]

def deltrees(dir , bases):
    [shutil.rmtree(f'{dir}/{base}') for base in bases]

def file_modified_date(path : Path | str , default = 19970101):
    if Path(path).exists():
        return int(time.strftime('%Y%m%d',time.localtime(os.path.getmtime(path))))
    else:
        return default

def file_modified_time(path : Path | str , default = 19970101000000):
    if Path(path).exists():
        return int(time.strftime('%Y%m%d%H%M%S',time.localtime(os.path.getmtime(path))))
    else:
        return default
