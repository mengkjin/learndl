import tushare as ts
from dataclasses import dataclass

_server_down = False
_token = '08232196bbac2a4ec51587784169f501f6387de3ad2fbe7719b5444a'
@dataclass
class TushareParams:
    """
    parameters for tushare
        server_down: whether the tushare server is down
        token: token for tushare
        pro: tushare pro api
    """
    server_down : bool = _server_down

    def __post_init__(self):
        self.pro = self.get_pro_api()
        
    def get_pro_api(self):
        return ts.pro_api(_token)

TS_PARAMS = TushareParams()