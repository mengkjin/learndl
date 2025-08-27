from typing import Literal
from src.basic import path as PATH

def load(conf_type : Literal['glob' , 'schedule' , 'factor' , 'boost' , 'nn' , 'train' , 'trade'] , name : str , **kwargs):
    p = PATH.conf.joinpath(conf_type , f'{name}.yaml')
    assert p.exists() , p
    return PATH.read_yaml(p , **kwargs)

def glob(name : str):
    if name == 'tushare_indus': kwargs = {'encoding' : 'gbk'}
    else: kwargs = {}
    return load('glob' , name , **kwargs)

def factor(name : str):
    return load('factor' , name)

def schedule(name : str):
    return load('schedule' , name)

def trade(name : str):
    return load('trade' , name)
