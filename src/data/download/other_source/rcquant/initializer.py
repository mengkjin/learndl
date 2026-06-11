"""
RiceQuant (rqdatac) initializer.
Call RcQuantInitializer.init to initialize the RiceQuant API.
"""
from __future__ import annotations
import rqdatac , re

from rqdatac.share.errors import QuotaExceeded

from src.proj import PATH , MACHINE , Logger , Base
from src.proj.util.catcher import IOCatcher

RQ_PATH = PATH.miscel.joinpath('Rcquant')

class MinDataType(Base.StrEnum):
    SEC = 'sec'
    ETF = 'etf'
    FUT = 'fut'
    CB = 'cb'
    
    @property
    def instrument(self) -> str:
        return {'sec' : 'CS' , 'etf' : 'ETF' , 'fut' : 'Future' , 'cb' : 'Convertible'}[self.value]

class RQInitializer:
    @classmethod
    def init(cls) -> bool:
        if not rqdatac.initialized(): 
            try:
                with IOCatcher() as catcher:
                    rqdatac.init(uri = MACHINE.secret.get('accounts' , 'rcquant/uri'))
                output = catcher.contents
                if _print := output['stdout']:
                    Logger.stdout(_print)
                if _error := output['stderr']:
                    key_info = re.search(r'Your account will be expired after  (\d+) days', _error)
                    if key_info:
                        Logger.alert1(f'RcQuant Warning >> {key_info.group(0)}')
                    else:
                        Logger.error(f'RcQuant Error : {_error}')
            except KeyError:
                Logger.error(f'rcquant login info not found, please check .secret/accounts.yaml')
                return False
            except QuotaExceeded as e:
                Logger.error(f'rcquant init failed: {e}')
                return False
        return True
