'''
basic variables in factor package
'''

from . import path as PATH

RISK_STYLE = [
    'size','beta','momentum','residual_volatility','non_linear_size',
    'book_to_price','liquidity','earnings_yield','growth','leverage'
]
RISK_INDUS = [
    'petro', 'coal', 'nonferrous', 'utility', 'public', 'steel', 'chemical', 'construct', 
    'cement', 'material', 'light', 'machine', 'power', 'defense', 'auto', 'retail', 'leisure', 
    'appliance', 'textile', 'health', 'liqor', 'food', 'agro', 'bank', 'financial', 'estate', 
    'transport', 'marine', 'airline', 'electronic', 'telecom', 'hardware', 'software', 'media', 'complex'
]

RISK_COMMON = ['market'] + RISK_INDUS + RISK_STYLE
AVAIL_BENCHMARKS = ['csi300' , 'csi500' , 'csi800' , 'csi1000'] 
DEFAULT_BENCHMARKS = ['csi300' , 'csi500' , 'csi1000']

EPS_WEIGHT = 1e-6
EPS_ACCURACY = 2e-6

TRADE_COST = 0.002

ROUNDING_EXPOSURE = 6
ROUNDING_CONTRIBUTION = 8
ROUNDING_RETURN = 8
ROUNDING_TURNOVER = 8

SYMBOL_INF = 0.0
SYMBOL_STOCK_LB = -1.
SYMBOL_STOCK_UB = +1.

DEFAULT_SOLVER_CONFIG : dict[str,dict] = {
    'cvxopt': {'show_progress': False}
}

DEFAULT_OPT_CONFIG = PATH.read_yaml(f'{PATH.conf}/fmp/default.yaml')
DEFAULT_OPT_CONFIG_2 : dict[str,dict] = {
    'equity' :  {
        'target_position' : 1. , # only remnant
        'target_value'    : None ,
        'add_position'    : None ,
        'add_value'       : None ,
    }, 
    'benchmark': {
        'benchmark': 'csi800' ,
    } ,
    'utility' : {
        'lambda' : 200. , 
        'trade_cost_rho'   : 0.01 ,
    } , 
    'pool' : {
        'basic'    : None ,    # basic pool , if none means estuniv
        'allow'    : None ,    # additional basic pool , and exempted from prohibit
        'ignore'   : None ,    # deal for opposite trades , similar as halt
        'no_sell'  : None ,    # not for sell
        'no_buy'   : None ,    # not for buy
        'prohibit' : None ,    # will sell if can , will not buy
        'warning'  : None ,    # will buy up to 0.5%
        'no_ldev'  : None ,    # not for under bought
        'no_udev'  : None ,    # not for over bought
        'shortable': None ,    # available for short
    } , 
    'induspool'  : {
        'no_sell'  : None ,    
        'no_buy'   : None ,
        'no_net_sell' : None , 
        'no_net_buy'  : None , 
    } , 
    'range' : {
        'ffmv.abs' : [None , None] ,   
        'ffmv.pct' : [0.1  , None] ,     
        'cp.abs'   : [None , None] ,
        'cp.pct'   : [None , None] ,
        #'amt.abs'  : [None , None] ,    
        #'amt.pct'  : [0.1  , None] ,    
        #'bv.abs'   : [None , None] ,   
        #'bv.pct'   : [0.1  , None] ,    
    } ,
    'limitation'  : {
        'no_st'    : True ,
        'list_days' : 63 ,
        'ignore_spec_risk' : False , 
        'te_constraint' : None ,
        'kcb_no_buy' : False ,
        'kcb_no_sell' : False ,
    } , 
    'bound'  : {
        'abs' : [-0.01 , 0.1] ,
        'rel' : [-0.01 , 0.01] ,
        'por' : [None , None] ,
    } , 
    'board'  : {
        'shse' : [None , None] , 
        'szse' : [None , None] , 
        'kcb'  : [None , None] , 
        'csi'  : [None , None]
    } , 
    'industry'  : {
        'abs' : [None , None] , 'rel' : [-0.01 , 0.01] , 'por' : [None , None] ,
    } , 
    'style' : {
        'abs' : [None , None] , 'rel' : [-0.5 , 0.5] , 'por' : [None , None] ,
        'size.rel' : [-0.25 , 0.25] ,
        'beta.rel' : [-0.25 , 0.25] ,
    } , 
    'component'  : {
        'component' : [-0.4 , None] , 
        'bsizedev1' : [None , None] , 
        'bsizedev2' : [None , None] ,
    } , 
    'turnover'  : {
        'single' : 0.2 , 
        'double' : None , 
    } , 
    'short' : {
        'short_position' : None ,
        'short_cost'     : None ,
    } , 
}