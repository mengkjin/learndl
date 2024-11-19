import os , shutil , yaml , time
from pathlib import Path

from ..project import MAIN_PATH

# variables

FACTOR_DESTINATION_LAPTOP = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha')
FACTOR_DESTINATION_SERVER = None # MAIN_PATH.joinpath('results' , 'Alpha')

main        = MAIN_PATH
data        = main.joinpath('data')
database    = data.joinpath('DataBase')

interim     = data.joinpath('Interim')
block       = interim.joinpath('DataBlock')
batch       = interim.joinpath('MiniBatch')
dataset     = interim.joinpath('DataSet')
norm        = interim.joinpath('HistNorm')

updater     = data.joinpath('Updater')

export      = data.joinpath('Export')
hidden      = export.joinpath('hidden_feature')
factor      = export.joinpath('stock_factor')
preds       = export.joinpath('model_prediction')

miscel      = data.joinpath('Miscellaneous')

conf        = main.joinpath('configs')
logs        = main.joinpath('logs')
model       = main.joinpath('models')
result      = main.joinpath('results')
boardsql    = main.joinpath('board_sqls')

def read_yaml(yaml_file , **kwargs):
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
