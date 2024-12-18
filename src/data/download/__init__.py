from .tushare import TushareDataDownloader
from .other_source import OtherSourceDownloader
from .sellside import SellsideSQLDownloader , SellsideFTPDownloader
from .hfm import JSDataUpdater

class CoreDataUpdater:
    @classmethod
    def update(cls):
        TushareDataDownloader.update()
        OtherSourceDownloader.update()

class SellsideDataUpdater:
    @classmethod
    def update(cls):
        SellsideSQLDownloader.update()
        SellsideFTPDownloader.update()
