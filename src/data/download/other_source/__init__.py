from .rcquant import RcquantMinBarDownloader
from .baostock_5m import Baostock5minBarDownloader

class OtherSourceDownloader:
    @classmethod
    def update(cls , * , indent: int = 0, vb_level: int = 1):
        RcquantMinBarDownloader.update(indent=indent, vb_level=vb_level)
        Baostock5minBarDownloader.update(indent=indent, vb_level=vb_level)
