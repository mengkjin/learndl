import tushare as ts
from src.proj import MACHINE

_server_down = False
class TushareConnector:
    """
    parameters for tushare
        server_down: whether the tushare server is down
        token: token for tushare
        pro: tushare pro api
    """
    server_down : bool = _server_down

    @property
    def token(self):
        return MACHINE.secrets['accounts']['tushare']['token']

    @property
    def pro(self):
        if not hasattr(self , '_pro'):
            self._pro = ts.pro_api(self.token)
        return self._pro
        
    def get_api(self):
        return ts.pro_api(self.token)

TS = TushareConnector()