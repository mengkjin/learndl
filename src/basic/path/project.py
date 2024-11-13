import os , shutil , sys , yaml , socket , torch
from pathlib import Path

# variables
THIS_IS_SERVER  = torch.cuda.is_available() and socket.gethostname() == 'mengkjin-server'
main = Path('D:/Coding/learndl/learndl') if not THIS_IS_SERVER else Path('home/mengkjin/Workspace/learndl')
sys.path.append(str(main))

FACTOR_DESTINATION_LAPTOP = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha')
FACTOR_DESTINATION_SERVER = None # MAIN_PATH.joinpath('results' , 'Alpha')

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
