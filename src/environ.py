import os , shutil

from dataclasses import dataclass

_src_dir = os.path.dirname(os.path.abspath(__file__))
_main_dir = os.path.dirname(_src_dir)
@dataclass
class CustomDirSpace:
    main  : str = _main_dir
    data  : str = f'{_main_dir}/data'
    conf  : str = f'{_main_dir}/configs'
    logs  : str = f'{_main_dir}/logs'
    model : str = f'{_main_dir}/model'
    instance : str = f'{_main_dir}/instance'
    result : str = f'{_main_dir}/result'

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