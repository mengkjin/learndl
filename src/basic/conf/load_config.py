from typing import Literal
from .. import path as PATH

def load(conf_type : Literal['glob' , 'confidential' , 'schedule' , 'factor' , 'boost' , 'nn' , 'train'] , name : str , **kwargs):
    p = PATH.conf.joinpath(conf_type , f'{name}.yaml')
    assert p.exists() , p
    return PATH.read_yaml(p , **kwargs)

def glob(name : str):
    if name == 'tushare_indus': kwargs = {'encoding' : 'gbk'}
    else: kwargs = {}
    return load('glob' , name , **kwargs)

def confidential(name : str):
    return load('confidential' , name)

def factor(name : str):
    return load('factor' , name)

def schedule(name : str):
    return load('schedule' , name)