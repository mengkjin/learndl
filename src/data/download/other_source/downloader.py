from __future__ import annotations
from src.proj import Base
from .rcquant import RcquantMinBarDownloader
from .baostock_5m import Baostock5minBarDownloader

class OtherSourceDownloader:
    @classmethod
    def update(cls , * , indent: int = 0, vb_level: int = 1) -> Base.UpdateFlagList:
        flags = Base.UpdateFlagList()
        flags += RcquantMinBarDownloader.update(indent=indent, vb_level=vb_level)
        flags += Baostock5minBarDownloader.update(indent=indent, vb_level=vb_level)
        return flags
