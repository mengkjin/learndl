"""
RiceQuant (rqdatac) initializer.
Call RcQuantInitializer.init to initialize the RiceQuant API.
"""
import rqdatac , re

from rqdatac.share.errors import QuotaExceeded
from typing import Literal

from src.proj import PATH , MACHINE , BaseClass
from src.proj.util import IOCatcher

RC_PATH = PATH.miscel.joinpath('Rcquant')
DATA_TYPES = Literal['sec' , 'etf' , 'fut' , 'cb']
instrument_types = {'sec' : 'CS' , 'etf' : 'ETF' , 'fut' : 'Future' , 'cb' : 'Convertible'}

class RcQuantInitializer(BaseClass.BoundLogger):
    @classmethod
    def init(cls) -> bool:
        if not rqdatac.initialized(): 
            try:
                with IOCatcher() as catcher:
                    rqdatac.init(uri = MACHINE.secret.get('accounts' , 'rcquant/uri'))
                output = catcher.contents
                if _print := output['stdout']:
                    cls.logger.stdout(_print)
                if _error := output['stderr']:
                    key_info = re.search(r'Your account will be expired after  (\d+) days', _error)
                    if key_info:
                        cls.logger.alert1(f'RcQuant Warning >> {key_info.group(0)}' , idt = 1)
                    else:
                        cls.logger.error(f'RcQuant Error : {_error}')
            except KeyError:
                cls.logger.error(f'rcquant login info not found, please check .secret/accounts.yaml')
                return False
            except QuotaExceeded as e:
                cls.logger.error(f'rcquant init failed: {e}')
                return False
        return True
