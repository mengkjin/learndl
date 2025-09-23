# basic variables in factor package
from src.proj import MACHINE

from typing import Literal , Any

UPDATE : dict[Literal['start' , 'end' , 'step'] , int] = {
    'start' : 20110101 if MACHINE.server else 20241101 ,
    'end'   : 20401231 if MACHINE.server else 20241231 ,
    'step'  : 5 ,
}

RISK : dict[Literal['market' , 'style' , 'indus'] ,list[str]] = MACHINE.configs('factor' , 'risk_factors')
BENCH : dict[Literal['availables' , 'defaults' , 'categories'] ,list[str]] = MACHINE.configs('factor' , 'benchmarks')

TRADE : dict[Literal['cost'] , dict[str , Any]] = MACHINE.configs('factor' , 'factor_util_params')['trade']
ROUNDING : dict[str , int] = MACHINE.configs('factor' , 'factor_util_params')['rounding']

DEFAULT_OPT_CONFIG : dict[str , Any] = MACHINE.configs('factor' , 'default_opt_config')
CUSTOM_OPT_CONFIG : dict[str , Any] = MACHINE.configs('factor' , 'custom_opt_config')

FACTOR_INIT_DATE : int = 20110101
CATEGORY0_SET : list[str] = ['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative']
CATEGORY1_SET : dict[str , list[str] | None] = {
    'fundamental' : ['quality' , 'growth' , 'value' , 'earning'] ,
    'analyst' : ['surprise' , 'coverage' , 'forecast' , 'adjustment'] ,
    'high_frequency' : ['hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity'] ,
    'behavior' : ['momentum' , 'volatility' , 'correlation' , 'liquidity'] ,
    'money_flow' : ['holding' , 'trading'] ,
    'alternative' : None
}

def Category0_to_Category1(category0 : str) -> list[str] | None:
    """Get the possible category1 of the category0 of stock factor"""
    return CATEGORY1_SET[category0]

def Category1_to_Category0(category1 : str) -> str:
    """Get the category0 given category1 of stock factor"""
    match category1:
        case 'quality' | 'growth' | 'value' | 'earning':
            return 'fundamental'
        case 'surprise' | 'coverage' | 'forecast' | 'adjustment':
            return 'analyst'
        case 'hf_momentum' | 'hf_volatility' | 'hf_correlation' | 'hf_liquidity':
            return 'high_frequency'
        case 'momentum' | 'volatility' | 'correlation' | 'liquidity':
            return 'behavior'
        case 'holding' | 'trading':
            return 'money_flow'
        case _:
            raise ValueError(f'undefined category1: {category1}')

class CategoryError(Exception): ...

def Validate_Category(category0 : str , category1 : str):
    if category0 not in CATEGORY0_SET:
        raise CategoryError(f'category0 is should be in {CATEGORY0_SET}, but got {category0}')

    if not category1:
        raise CategoryError('category1 is not set')

    if (category1_list := CATEGORY1_SET[category0]):
        if category1 not in category1_list:
            raise CategoryError(f'category1 is should be in {category1_list}, but got {category1}')
       