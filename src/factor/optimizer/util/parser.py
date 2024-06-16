from copy import deepcopy
from typing import Any

from src.environ import RISK_INDUS , RISK_STYLE
from .bound import StockBound , StockPool , IndustryPool , GeneralBound , ValidRange , STOCK_UB , STOCK_LB


def parse_config_equity(config : dict) -> dict[str,float|Any]:
    '''
    Total Position, take 4 in , 1 out:
    config:
        in  : 'target_position' , 'target_value' , 'add_position' , 'add_value'
        out : 'target_position'
    init_value: necessary if 'target_value' , 'add_value' applies
    init_port:  necessary if 'add_position' , 'add_value' applies
    '''
    config = config['equity']
    assert sum([v is not None for v in config.values()]) == 1 , f'must be exactly 1 non NA option in {config}'
    return config

def parse_config_benchmark(config : dict) -> dict[str,Any]:
    '''Benchmark, return raw config'''
    config = config['benchmark']
    return config

def parse_config_utility(config : dict) -> dict[str,float|None]:
    '''Utility, return raw config'''
    config = config['utility']
    return config

def parse_config_pool(config : dict) -> StockPool:
    '''
    Stock Pool, take various list_like , 1 StockPool out:
    config : 
        in : 'basic', 'allow' , 'ignore' , 'no_sell' , 'no_buy' , 'prohibit' , 'warning' , 'no_ldev', 'no_udev' 
        out: StockPool
    '''
    config = config['pool']
    return StockPool(**config)

def parse_config_induspool(config : dict) -> IndustryPool:
    '''
    Indus Pool, take various list_like , various list_like out , eleminate empty:
    config : 
        in : 'no_sell', 'no_buy' , 'no_net_sell' , 'no_net_buy' 
    '''
    config = config['induspool']
    return IndustryPool(**config)

def parse_config_range(config : dict) -> dict[str,ValidRange]:
    '''
    Ilimitations, take various miscellaneous in , various miscellaneous out , eleminate empty:
    config : 
        in : 'ffmv.abs', 'ffmv.pct' , 'cp.abs' , 'cp.pct'
        future: 'amt.abs' , 'amt.pct' , 'bv.abs' , 'bv.pct'
    '''
    config = deepcopy(config['range'])
    for rngtype in ['abs' , 'pct']:
        for valname in ['ffmv' , 'cp' , 'amt' , 'bv']:
            if f'{valname}.{rngtype}' in config.keys():
                config[f'{valname}.{rngtype}'] = ValidRange(rngtype , *config[f'{valname}.{rngtype}'])
    return {k:v for k,v in config.items() if v}

def parse_config_limitation(config : dict) -> dict[str,Any]:
    '''
    Ilimitations, take various miscellaneous in , various miscellaneous out:
    config : 
        in : 'no_st' , 'list_days' , 'kcb_no_buy' , 'kcb_no_sell' 'ignore_spec_risk' , 'te_constraint'
    '''
    config = config['limitation']
    return config

def parse_config_bound(config : dict) -> dict[str,GeneralBound]:
    '''
    Stock Overall Bound, take 3 list in , 3 GeneralBound out , eleminate empty:
    config : 
        in : 'abs', 'rel' , 'por'
    '''
    config = config['bound']
    return {key:bnd for key,bnd in {k:GeneralBound(k , *v) for k,v in config.items()}.items() if bnd}

def parse_config_board(config : dict):
    '''
    Board Absolute Bound, take various list in , various GeneralBound out , eleminate empty:
    config : 
        in : 'shse', 'szse' , 'kcb' , 'csi'
    '''
    default_bnd_key = 'abs' # 'rel' , 'por'
    config = config['board']
    rslt : dict[str,list[GeneralBound]] = {}
    for key , val in config.items():
        s = key.split('.')
        name , bnd_key = (s[0] , default_bnd_key) if len(s) == 1 else s
        if bnd := GeneralBound(bnd_key , *val):
            if not name in rslt: rslt[name] = []
            rslt[name].append(bnd)
    return rslt

def parse_config_industry(config : dict):
    '''
    Industry Bound, take various list in , various GeneralBound out , eleminate empty:
    config : 
        in : 'abs', 'rel' , 'por' , '{indus}.abs', '{indus}.rel' , '{indus}.por'
    '''
    default_bnd_key = 'rel' # 'rel' , 'por'
    config = config['industry']
    rslt : dict[str,list[GeneralBound]] = {}
    overall_bnds = [GeneralBound(bnd_key , *config[bnd_key]) for bnd_key in ['abs' , 'rel' , 'por'] if bnd_key in config]
    overall_bnds = [bnd for bnd in overall_bnds if bnd]
    for name in RISK_INDUS:
        sub_config = {key:val for key , val in config.items() if key.startswith(name)}
        if sub_config:
            bnds = [GeneralBound(default_bnd_key if key == name else key.split('.')[1] , *val) for key , val in sub_config.items()]
            rslt[name] = bnds
        else:
            rslt[name] = overall_bnds
    return rslt

def parse_config_style(config : dict):
    '''
    Style Bound, take various list in , various GeneralBound out , eleminate empty:
    config : 
        in : 'abs', 'rel' , 'por' , '{style}.abs', '{style}.rel' , '{style}.por'
    '''
    default_bnd_key = 'rel' # 'rel' , 'por'
    config = config['style']
    rslt : dict[str,list[GeneralBound]] = {}
    overall_bnds = [GeneralBound(bnd_key , *config[bnd_key]) for bnd_key in ['abs' , 'rel' , 'por'] if bnd_key in config]
    overall_bnds = [bnd for bnd in overall_bnds if bnd]
    for name in RISK_STYLE:
        sub_config = {key:val for key , val in config.items() if key.startswith(name)}
        if sub_config:
            bnds = [GeneralBound(default_bnd_key if key == name else key.split('.')[1] , *val) for key , val in sub_config.items()]
            rslt[name] = bnds
        else:
            rslt[name] = overall_bnds
    return rslt

def parse_config_component(config : dict):
    '''
    Component Relative Bound, take various list in , various GeneralBound out , eleminate empty:
    config : 
        in : 'component', 'bsizedev1' , 'bsizedev2'
    '''
    default_bnd_key = 'rel' # 'rel' , 'por'
    config = config['component']
    rslt : dict[str,list[GeneralBound]] = {}
    for key , val in config.items():
        s = key.split('.')
        name , bnd_key = (s[0] , default_bnd_key) if len(s) == 1 else s
        if bnd := GeneralBound(bnd_key , *val):
            if not name in rslt: rslt[name] = []
            rslt[name].append(bnd)
    return rslt

def parse_config_turnover(config : dict) -> dict[str,float|None]:
    '''
    Turnover upper Bound, take 2 float | None in , 1 float | None out:
    config : 
        in : 'single', 'double'
        out: 'double
    '''
    config = config['turnover']
    value1 , value2 = config['single'] , config['double']
    if value1 and value2: value = min(value1 * 2 , value2)
    elif value1: value = value1 * 2
    else: value = value2
    return {'double' : value}

def parse_config_short(config : dict) -> dict[str,float|Any]:
    '''
    shortsell constraint, take 2 miscellaneous in , 2 miscellaneous out:
    config : 
        in : 'short_position', 'short_cost'
    '''
    return config['short'] 
