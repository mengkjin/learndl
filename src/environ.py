import os , shutil

_current_dir = os.path.dirname(os.path.abspath(__file__)) 
DIR_main = f'{_current_dir}/..'
DIR_data = f'{_current_dir}/../data'
DIR_conf = f'{_current_dir}/../configs'
DIR_logs = f'{_current_dir}/../logs'


def rmdir(d , remake_dir = False):
    if isinstance(d , (list,tuple)):
        [shutil.rmtree(x) for x in d if os.path.exists(x)]
        if remake_dir : [os.makedirs(x , exist_ok = True) for x in d]
    elif isinstance(d , str):
        if os.path.exists(d): shutil.rmtree(d)
        if remake_dir : os.mkdir(d)
    else:
        raise Exception(f'KeyError : {str(d)}')