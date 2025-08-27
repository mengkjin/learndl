from .rcquant import RcquantMinBarDownloader
from .baostock_5m import Baostock5minBarDownloader

class OtherSourceDownloader:
    @classmethod
    def update(cls):
        RcquantMinBarDownloader.proceed()
        Baostock5minBarDownloader.proceed()
