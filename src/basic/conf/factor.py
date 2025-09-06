'''
basic variables in factor package
'''
from src.proj import MACHINE

from typing import Literal , Any

UPDATE = {
    'start' : 20110101 if MACHINE.server else 20241101 ,
    'end'   : 20401231 if MACHINE.server else 20241231 ,
    'step'  : 5 ,
}

RISK : dict[Literal['market' , 'style' , 'indus'] ,list[str]] = MACHINE.configs('factor' , 'risk_factors')
BENCH : dict[Literal['availables' , 'defaults' , 'categories'] ,list[str]] = MACHINE.configs('factor' , 'benchmarks')

_factor_util_params = MACHINE.configs('factor' , 'factor_util_params')
TRADE : dict[Literal['cost'] , dict[str , Any]] = _factor_util_params['trade']
ROUNDING : dict[str , int] = _factor_util_params['rounding']

DEFAULT_OPT_CONFIG = MACHINE.configs('factor' , 'default_opt_config')
CUSTOM_OPT_CONFIG = MACHINE.configs('factor' , 'custom_opt_config')