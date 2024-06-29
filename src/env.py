import socket , torch
from .env_util import _RegModel , _CustomConf , _CustomPath

BOOSTER_MODULE = ['lgbm']
THIS_IS_SERVER = torch.cuda.is_available() # socket.gethostname() == 'mengkjin-server'
assert not THIS_IS_SERVER or torch.cuda.is_available() , f'SERVER must have cuda available'
    
REG_MODELS = [
    _RegModel('gru_day'    , 'swalast' , 0 , 'gru_day_V0') ,
    _RegModel('gruRTN_day' , 'swalast' , 0 , 'gruRTN_day_V0') , 
    _RegModel('gruRES_day' , 'swalast' , 0 , 'gruRES_day_V0') ,
]
FACTOR_DESTINATION = '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha'

PATH = _CustomPath()
CONF = _CustomConf()

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
