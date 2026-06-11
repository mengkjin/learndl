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
from __future__ import annotations

from src.proj import Base
from src.data.download import (
    TushareDataDownloader , OtherSourceDownloader , SellsideSQLDownloader , SellsideFTPDownloader
)
from .custom import BasicCustomUpdater
from .hfm import JSDataUpdater
__all__ = ['CoreDataUpdater' , 'SellsideDataUpdater' , 'JSDataUpdater' , 'CustomDataUpdater']

class CoreDataUpdater:
    """Orchestrate updates for core market data (Tushare + other sources)."""
    @classmethod
    def update(cls) -> Base.UpdateFlagList:
        """Run incremental updates for Tushare and other data sources."""
        flags = Base.UpdateFlagList()
        flags += TushareDataDownloader.update()
        flags += OtherSourceDownloader.update()
        return flags

    @classmethod
    def rollback(cls , rollback_date : int) -> Base.UpdateFlagList:
        """Rollback Tushare data to ``rollback_date``."""
        flags = Base.UpdateFlagList()
        flags += TushareDataDownloader.rollback(rollback_date)
        return flags

class SellsideDataUpdater:
    """Orchestrate updates for sell-side data (SQL and FTP sources)."""
    @classmethod
    def update(cls) -> Base.UpdateFlagList:
        """Run incremental updates for sell-side SQL and FTP data sources."""
        flags = Base.UpdateFlagList()
        flags += SellsideSQLDownloader.update()
        flags += SellsideFTPDownloader.update()
        return flags

    @classmethod
    def rollback(cls , rollback_date : int) -> Base.UpdateFlagList:
        """Rollback sell-side data (not yet implemented)."""
        flags = Base.UpdateFlagList()
        return flags

class CustomDataUpdater:
    """Orchestrate updates for all registered ``BasicCustomUpdater`` subclasses."""
    @classmethod
    def update(cls) -> Base.UpdateFlagList:
        """call ``update()`` on each updater"""
        flags = Base.UpdateFlagList()
        for updater in BasicCustomUpdater.iter_updaters():
            flags += updater.update(indent = 1 , vb_level = 2)
        return flags

    @classmethod
    def rollback(cls , rollback_date : int) -> Base.UpdateFlagList:
        """call ``rollback()`` on each updater"""
        flags = Base.UpdateFlagList()
        for updater in BasicCustomUpdater.iter_updaters():
            flags += updater.rollback(rollback_date , indent = 1 , vb_level = 2)
        return flags