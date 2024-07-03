import os , shutil , yaml
from dataclasses import dataclass
from typing import Literal , Optional

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(_SRC_DIR)

@dataclass
class CustomPath:
    main        : str = _MAIN_DIR
    data        : str = os.path.join(_MAIN_DIR , 'data')
    batch       : str = os.path.join(_MAIN_DIR , 'data' , 'MiniBatch')
    block       : str = os.path.join(_MAIN_DIR , 'data' , 'DataBlock')
    database    : str = os.path.join(_MAIN_DIR , 'data' , 'DataBase')
    dataset     : str = os.path.join(_MAIN_DIR , 'data' , 'DataSet')
    norm        : str = os.path.join(_MAIN_DIR , 'data' , 'HistNorm')
    tree        : str = os.path.join(_MAIN_DIR , 'data' , 'TreeData')
    updater     : str = os.path.join(_MAIN_DIR , 'data' , 'Updater')
    conf        : str = os.path.join(_MAIN_DIR , 'configs')
    logs        : str = os.path.join(_MAIN_DIR , 'logs')
    model       : str = os.path.join(_MAIN_DIR , 'model')
    result      : str = os.path.join(_MAIN_DIR , 'result')

    def __post_init__(self):
        [os.makedirs(v , exist_ok=True) for v in self.__dict__.values()]

    @staticmethod
    def read_yaml(yaml_file):
        with open(yaml_file ,'r') as f:
            d = yaml.load(f , Loader = yaml.FullLoader)
        return d
    
    @staticmethod
    def dump_yaml(data , yaml_file):
        with open(yaml_file , 'a' if os.path.exists(yaml_file) else 'w') as f:
            yaml.dump(data , f)

    @staticmethod
    def copytree(src , dst):
        shutil.copytree(src , dst)

    @staticmethod
    def copyfiles(src , dst , bases):
        [shutil.copyfile(f'{src}/{base}' , f'{dst}/{base}') for base in bases]

    @staticmethod
    def deltrees(dir , bases):
        [shutil.rmtree(f'{dir}/{base}') for base in bases]

PATH = CustomPath()