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

from collections.abc import Sequence
from typing import Any

from src.proj import Base
from src.data.download import (
    TushareDataDownloader , OtherSourceDownloader , SellsideSQLDownloader , # SellsideFTPDownloader
)
from .custom import BasicCustomUpdater
from .hfm import JSDataUpdater
__all__ = ['CoreDataUpdater' , 'SellsideDataUpdater' , 'JSDataUpdater' , 'CustomDataUpdater']

class CoreDataUpdater(Base.SelectiveUpdateSupport):
    """Orchestrate updates for core market data (Tushare + other sources)."""

    @classmethod
    def selection_tree(cls) -> Base.UpdateMenuNode:
        return Base.UpdateMenuNode(
            label = 'Core' ,
            help = 'Tushare market/fina data and other minute-bar sources' ,
            children = [
                TushareDataDownloader.selection_tree() ,
                OtherSourceDownloader.selection_tree() ,
            ] ,
        )

    @classmethod
    def selective_update(
        cls ,
        selection : Sequence[str] ,
        * ,
        force : bool = False ,
        start : int | None = None ,
        end : int | None = None ,
        **kwargs : Any ,
    ) -> Base.UpdateFlag:
        tushare_keys = [key for key in selection if key.startswith(f'{TushareDataDownloader.SELECTION_PREFIX}.')]
        other_keys = [key for key in selection if key.startswith(f'{OtherSourceDownloader.SELECTION_PREFIX}.')]
        flags = Base.UpdateFlagList()
        if tushare_keys:
            flags += TushareDataDownloader.selective_update(
                tushare_keys , force = force , start = start , end = end , **kwargs)
        if other_keys:
            flags += OtherSourceDownloader.selective_update(
                other_keys , force = force , start = start , end = end , **kwargs)
        return flags.summarize() if flags else Base.UpdateFlag.SKIPPED

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

class SellsideDataUpdater(Base.SelectiveUpdateSupport):
    """Orchestrate updates for sell-side data (SQL and FTP sources)."""
    SELECTION_PREFIX = 'sellside'

    @classmethod
    def selection_tree(cls) -> Base.UpdateMenuNode:
        from collections import defaultdict
        groups : dict[str , list[str]] = defaultdict(list)
        for key in SellsideSQLDownloader.available_factors():
            vendor = key.split('.' , 1)[0]
            groups[vendor].append(key)
        children : list[Base.UpdateMenuNode] = []
        for vendor in sorted(groups):
            leaves = [
                Base.UpdateMenuNode(
                    label = factor_key ,
                    key = f'{cls.SELECTION_PREFIX}.{factor_key}' ,
                    help = f'Sellside SQL factor {factor_key}' ,
                )
                for factor_key in sorted(groups[vendor])
            ]
            children.append(Base.UpdateMenuNode(
                label = vendor ,
                children = leaves ,
                help = f'Sellside factors from {vendor}' ,
            ))
        return Base.UpdateMenuNode(
            label = 'Sellside' ,
            children = children ,
            help = 'Sell-side SQL factor downloaders' ,
        )

    @classmethod
    def selective_update(
        cls ,
        selection : Sequence[str] ,
        * ,
        force : bool = False ,
        start : int | None = None ,
        end : int | None = None ,
        **kwargs : Any ,
    ) -> Base.UpdateFlag:
        prefix = f'{cls.SELECTION_PREFIX}.'
        keys = [key[len(prefix):] for key in selection if key.startswith(prefix)]
        if not keys:
            return Base.UpdateFlag.SKIPPED
        force , start , end = Base.resolve_force_range(force , start , end , key = 'sellside_sql')
        if force:
            assert start is not None and end is not None
            return SellsideSQLDownloader.update_dates(start , end , overwrite = True , keys = keys)
        return SellsideSQLDownloader.update_since(trace = 0 , keys = keys)

    @classmethod
    def update(cls) -> Base.UpdateFlagList:
        """Run incremental updates for sell-side SQL and FTP data sources."""
        flags = Base.UpdateFlagList()
        flags += SellsideSQLDownloader.update()
        # flags += SellsideFTPDownloader.update()
        return flags

    @classmethod
    def rollback(cls , rollback_date : int) -> Base.UpdateFlagList:
        """Rollback sell-side data (not yet implemented)."""
        flags = Base.UpdateFlagList()
        return flags

class CustomDataUpdater(Base.SelectiveUpdateSupport):
    """Orchestrate updates for all registered ``BasicCustomUpdater`` subclasses."""
    SELECTION_PREFIX = 'custom'

    @classmethod
    def selection_tree(cls) -> Base.UpdateMenuNode:
        BasicCustomUpdater.import_updaters()
        children : list[Base.UpdateMenuNode] = []
        for name , updater in sorted(BasicCustomUpdater.registry.items()):
            if name == 'CustomIndexUpdater':
                from .custom.custom_index import CustomIndex
                index_leaves = [
                    Base.UpdateMenuNode(
                        label = index_name ,
                        key = f'{cls.SELECTION_PREFIX}.{name}.{index_name}' ,
                        help = f'Custom index {index_name}' ,
                    )
                    for index_name in sorted(CustomIndex.registry)
                ]
                children.append(Base.UpdateMenuNode(
                    label = name ,
                    children = index_leaves or [
                        Base.UpdateMenuNode(
                            label = name ,
                            key = f'{cls.SELECTION_PREFIX}.{name}' ,
                            help = 'All custom indices' ,
                        )
                    ] ,
                    help = 'Custom index portfolios' ,
                ))
            else:
                children.append(Base.UpdateMenuNode(
                    label = name ,
                    key = f'{cls.SELECTION_PREFIX}.{name}' ,
                    help = f'Custom updater {name}' ,
                ))
        return Base.UpdateMenuNode(
            label = 'Custom' ,
            children = children ,
            help = 'Affiliated / label / risk feature updaters' ,
        )

    @classmethod
    def selective_update(
        cls ,
        selection : Sequence[str] ,
        * ,
        force : bool = False ,
        start : int | None = None ,
        end : int | None = None ,
        **kwargs : Any ,
    ) -> Base.UpdateFlag:
        BasicCustomUpdater.import_updaters()
        force , start , end = Base.resolve_force_range(force , start , end)
        flags = Base.UpdateFlagList()
        # Group custom index leaves under CustomIndexUpdater
        index_names : list[str] = []
        updater_names : set[str] = set()
        prefix = f'{cls.SELECTION_PREFIX}.'
        for key in selection:
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if rest.startswith('CustomIndexUpdater.'):
                index_names.append(rest.split('.' , 1)[1])
                updater_names.add('CustomIndexUpdater')
            else:
                updater_names.add(rest)

        for name in sorted(
            updater_names ,
            key = lambda n: (getattr(BasicCustomUpdater.registry[n] , 'UPDATE_ORDER' , 0) , n) ,
        ):
            updater = BasicCustomUpdater.registry[name]
            indent = kwargs.get('indent' , 0)
            vb_level = kwargs.get('vb_level' , 1)
            if force:
                assert start is not None and end is not None
                updater.SetClassVB(vb_level , indent)
                if name == 'CustomIndexUpdater' and index_names:
                    flags += updater.proceed_update(
                        start = start , end = end , overwrite = True , index_names = index_names)
                else:
                    flags += updater.proceed_update(start = start , end = end , overwrite = True)
            else:
                if name == 'CustomIndexUpdater' and index_names:
                    flags += updater.update(
                        index_names = index_names , indent = indent , vb_level = vb_level)
                else:
                    flags += updater.update(indent = indent , vb_level = vb_level)
        return flags.summarize() if flags else Base.UpdateFlag.SKIPPED

    @classmethod
    def update(cls) -> Base.UpdateFlagList:
        """call ``update()`` on each updater"""
        flags = Base.UpdateFlagList()
        for updater in BasicCustomUpdater.iter_updaters():
            flags += updater.update(indent = 0 , vb_level = 1)
        return flags

    @classmethod
    def rollback(cls , rollback_date : int) -> Base.UpdateFlagList:
        """call ``rollback()`` on each updater"""
        flags = Base.UpdateFlagList()
        for updater in BasicCustomUpdater.iter_updaters():
            flags += updater.rollback(rollback_date , indent = 0 , vb_level = 1)
        return flags
