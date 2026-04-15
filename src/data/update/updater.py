"""
Top-level data update orchestrators for the full pipeline.

Classes
-------
CoreDataUpdater
    Drives Tushare and other-source downloaders (market data, financial statements,
    minute bars from BaoStock/RiceQuant).
SellsideDataUpdater
    Drives sell-side SQL and FTP downloaders (Dongfang L2, broker factor data).
CustomDataUpdater
    Iterates over all registered ``BasicCustomUpdater`` subclasses (labels,
    daily risk features, multi-kline, custom indices, etc.).
"""
from .custom import BasicCustomUpdater
from .hfm import JSDataUpdater

from src.data.download import (
    TushareDataDownloader , OtherSourceDownloader , SellsideSQLDownloader , SellsideFTPDownloader
)

__all__ = ['CoreDataUpdater' , 'SellsideDataUpdater' , 'JSDataUpdater' , 'CustomDataUpdater']

class CoreDataUpdater:
    """Orchestrate updates for core market data (Tushare + other sources)."""
    @classmethod
    def update(cls):
        """Run incremental updates for Tushare and other data sources."""
        TushareDataDownloader.update()
        OtherSourceDownloader.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        """Rollback Tushare data to ``rollback_date``."""
        TushareDataDownloader.rollback(rollback_date)

class SellsideDataUpdater:
    """Orchestrate updates for sell-side data (SQL and FTP sources)."""
    @classmethod
    def update(cls):
        """Run incremental updates for sell-side SQL and FTP data sources."""
        SellsideSQLDownloader.update()
        SellsideFTPDownloader.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        """Rollback sell-side data (not yet implemented)."""
        ...

class CustomDataUpdater:
    """Orchestrate updates for all registered ``BasicCustomUpdater`` subclasses."""
    @classmethod
    def update(cls):
        """Import all custom updater modules and call ``update()`` on each."""
        BasicCustomUpdater.import_updaters()
        for name , updater in BasicCustomUpdater.registry.items():
            updater.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        """Import all custom updater modules and call ``rollback()`` on each."""
        BasicCustomUpdater.import_updaters()
        for name , updater in BasicCustomUpdater.registry.items():
            updater.rollback(rollback_date)