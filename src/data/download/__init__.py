from .tushare import TushareDownloader
from .other_source import OtherSourceDownloader
from .sellside import SQLDownloader

class DataDownloader:
    @classmethod
    def proceed(cls):
        TushareDownloader.proceed()
        OtherSourceDownloader.proceed()
        SQLDownloader.update_since()
