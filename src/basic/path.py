import os , shutil , yaml
from pathlib import Path

main        = [parent for parent in list(Path(__file__).parents) if parent.match('./src/')][-1].parent
data        = main.joinpath('data')
batch       = data.joinpath('MiniBatch')
block       = data.joinpath('DataBlock')
database    = data.joinpath('DataBase')
dataset     = data.joinpath('DataSet')
hidden      = data.joinpath('ModelHidden')
norm        = data.joinpath('HistNorm')
tree        = data.joinpath('TreeData')
updater     = data.joinpath('Updater')

conf        = main.joinpath('configs')
conf_train  = conf.joinpath('train')
conf_nn     = conf.joinpath('nn')
conf_boost  = conf.joinpath('boost')

logs        = main.joinpath('logs')
model       = main.joinpath('models')
result      = main.joinpath('results')

def read_yaml(yaml_file , **kwargs):
    if isinstance(yaml_file , Path):
        if yaml_file.is_dir() and yaml_file.joinpath('default.yaml').exists():
            yaml_file = yaml_file.joinpath('default.yaml')
        elif not yaml_file.exists() and yaml_file.suffix == '' and yaml_file.with_name(f'{yaml_file.name}.yaml').exists():
            yaml_file = yaml_file.with_name(f'{yaml_file.name}.yaml')
    with open(yaml_file ,'r' , **kwargs) as f:
        d = yaml.load(f , Loader = yaml.FullLoader)
    return d

def dump_yaml(data , yaml_file , **kwargs):
    with open(yaml_file , 'a' if os.path.exists(yaml_file) else 'w') as f:
        yaml.dump(data , f , **kwargs)

def setting(name : str):
    p = conf.joinpath('setting' , f'{name}.yaml')
    assert p.exists() , p
    kwargs = {}
    if name == 'tushare_indus': kwargs['encoding'] = 'gbk'
    return read_yaml(p , **kwargs)

def copytree(src , dst):
    shutil.copytree(src , dst)

def copyfiles(src , dst , bases):
    [shutil.copyfile(f'{src}/{base}' , f'{dst}/{base}') for base in bases]

def deltrees(dir , bases):
    [shutil.rmtree(f'{dir}/{base}') for base in bases]
