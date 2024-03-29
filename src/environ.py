import os , shutil

from dataclasses import dataclass

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SRC_DIR)

@dataclass
class CustomDirSpace:
    main        : str = MAIN_DIR
    data        : str = os.path.join(MAIN_DIR , 'data')
    block       : str = os.path.join(MAIN_DIR , 'data' , 'block_data')
    hist_norm   : str = os.path.join(MAIN_DIR , 'data' , 'hist_norm')
    torch_pack  : str = os.path.join(MAIN_DIR , 'data' , 'torch_pack')
    db          : str = os.path.join(MAIN_DIR , 'data' , 'DataBase')
    db_updater  : str = os.path.join(MAIN_DIR , 'data' , 'Updater')
    conf        : str = os.path.join(MAIN_DIR , 'configs')
    logs        : str = os.path.join(MAIN_DIR , 'logs')
    model       : str = os.path.join(MAIN_DIR , 'model')
    instance    : str = os.path.join(MAIN_DIR , 'instance')
    result      : str = os.path.join(MAIN_DIR , 'result')

    def __post_init__(self):
        [os.makedirs(v , exist_ok=True) for v in self.__dict__.values()]

DIR = CustomDirSpace()

def rmdir(d , remake_dir = False):
    if isinstance(d , (list,tuple)):
        [shutil.rmtree(x) for x in d if os.path.exists(x)]
        if remake_dir : [os.makedirs(x , exist_ok = True) for x in d]
    elif isinstance(d , str):
        if os.path.exists(d): shutil.rmtree(d)
        if remake_dir : os.mkdir(d)
    else:
        raise Exception(f'KeyError : {str(d)}')