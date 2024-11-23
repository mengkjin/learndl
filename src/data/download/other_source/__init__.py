from .rcquant import RcquantMinBarDownloader
from .baostock_5m import Baostock5minBarDownloader

class OtherSourceDownloader:
    @classmethod
    def proceed(cls):
        RcquantMinBarDownloader.proceed()
        Baostock5minBarDownloader.proceed()
