import os , shutil , yaml

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(os.path.dirname(_SRC_DIR))

main        : str = _MAIN_DIR
data        : str = os.path.join(_MAIN_DIR , 'data')
batch       : str = os.path.join(_MAIN_DIR , 'data' , 'MiniBatch')
block       : str = os.path.join(_MAIN_DIR , 'data' , 'DataBlock')
database    : str = os.path.join(_MAIN_DIR , 'data' , 'DataBase')
dataset     : str = os.path.join(_MAIN_DIR , 'data' , 'DataSet')
hidden      : str = os.path.join(_MAIN_DIR , 'data' , 'ModelHidden')
norm        : str = os.path.join(_MAIN_DIR , 'data' , 'HistNorm')
tree        : str = os.path.join(_MAIN_DIR , 'data' , 'TreeData')
updater     : str = os.path.join(_MAIN_DIR , 'data' , 'Updater')
conf        : str = os.path.join(_MAIN_DIR , 'configs')
logs        : str = os.path.join(_MAIN_DIR , 'logs')
model       : str = os.path.join(_MAIN_DIR , 'model')
result      : str = os.path.join(_MAIN_DIR , 'result')

for v in [main,data,batch,block,database,dataset,hidden,norm,tree,updater,conf,logs,model,result]:
    os.makedirs(v,exist_ok=True)

def read_yaml(yaml_file , **kwargs):
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
