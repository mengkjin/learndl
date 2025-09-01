from typing import Literal

from src.proj import PATH

def load(conf_type : Literal['glob' , 'registry' , 'factor' , 'boost' , 'nn' , 'train' , 'trade' , 'schedule'] , name : str , **kwargs):
    p = PATH.conf.joinpath(conf_type , f'{name}.yaml')
    assert p.exists() , p
    return PATH.read_yaml(p , **kwargs)

def glob(name : str):
    if name == 'tushare_indus': kwargs = {'encoding' : 'gbk'}
    else: kwargs = {}
    return load('glob' , name , **kwargs)

def factor(name : str):
    return load('factor' , name)

def registry(name : str):
    return load('registry' , name)

def trade(name : str):
    return load('trade' , name)

def schedule(name : str):
    return load('schedule' , name)

def local(name : str):
    p = PATH.local_settings.joinpath(f'{name}.yaml')
    assert p.exists() , p
    return PATH.read_yaml(p)
