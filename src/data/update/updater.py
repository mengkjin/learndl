from .custom import BasicCustomUpdater
from .hfm import JSDataUpdater

from src.data.download import (
    TushareDataDownloader , OtherSourceDownloader , SellsideSQLDownloader , SellsideFTPDownloader
)

__all__ = ['CoreDataUpdater' , 'SellsideDataUpdater' , 'JSDataUpdater' , 'CustomDataUpdater']

class CoreDataUpdater:
    @classmethod
    def update(cls):
        TushareDataDownloader.update()
        OtherSourceDownloader.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        TushareDataDownloader.rollback(rollback_date)

class SellsideDataUpdater:
    @classmethod
    def update(cls):
        SellsideSQLDownloader.update()
        SellsideFTPDownloader.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        ...

class CustomDataUpdater:
    @classmethod
    def update(cls):
        BasicCustomUpdater.import_updaters()
        for name , updater in BasicCustomUpdater.registry.items():
            updater.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        BasicCustomUpdater.import_updaters()
        for name , updater in BasicCustomUpdater.registry.items():
            updater.rollback(rollback_date)