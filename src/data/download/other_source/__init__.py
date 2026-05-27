from .rcquant import RcquantMinBarDownloader
from .baostock_5m import Baostock5minBarDownloader

from src.proj.util import BaseModule

class OtherSourceDownloader(BaseModule):
    @classmethod
    def update(cls , * , indent: int = 0, vb_level: int = 1):
        cls.SetClassVB(vb_level, indent)
        cls.logger.note('Download since last update!')
        RcquantMinBarDownloader.update(indent=indent, vb_level=vb_level)
        Baostock5minBarDownloader.update(indent=indent, vb_level=vb_level)
