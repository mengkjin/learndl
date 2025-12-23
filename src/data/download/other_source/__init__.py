from .rcquant import RcquantMinBarDownloader
from .baostock_5m import Baostock5minBarDownloader

from src.proj import Logger

class OtherSourceDownloader:
    @classmethod
    def update(cls):
        Logger.stdout(f'Download: {cls.__name__} since last update!')
        RcquantMinBarDownloader.proceed()
        Baostock5minBarDownloader.proceed()
