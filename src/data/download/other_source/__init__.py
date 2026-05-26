from .rcquant import RcquantMinBarDownloader
from .baostock_5m import Baostock5minBarDownloader

from src.proj import Logger

class OtherSourceDownloader:
    @classmethod
    def update(cls):
        Logger.note(f'{cls.__name__} : Download since last update!')
        RcquantMinBarDownloader.proceed()
        Baostock5minBarDownloader.proceed()
