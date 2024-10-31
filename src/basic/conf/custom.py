from .. import path as PATH

def glob(name : str):
    if name == 'tushare_indus': kwargs = {'encoding' : 'gbk'}
    else: kwargs = {}
    p = PATH.conf.joinpath('glob' , f'{name}.yaml')
    assert p.exists() , p
    return PATH.read_yaml(p , **kwargs)
