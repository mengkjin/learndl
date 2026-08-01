"""
Top-level orchestrator for all Tushare data fetchers.

``TushareDataDownloader`` iterates over all registered ``TushareFetcher``
subclasses (discovered via dynamic module import in ``TushareFetcher.load_tasks()``)
and calls their ``update()`` or ``rollback()`` methods.

``TSBackUpDataTransform.clear/update/rollback`` manages the manually-downloaded
CSV backup data that supplements the live Tushare pipeline.
"""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Generator , Sequence
from typing import Any

from src.proj import Base
from src.data.download.tushare.basic import TushareFetcher , TSBackUpDataTransform

__all__ = ['TushareDataDownloader']

class TushareDataDownloader(Base.BasicUpdater , Base.SelectiveUpdateSupport):
    """Orchestrate incremental updates for all registered Tushare fetchers."""
    UPDATE_ALIAS = 'download'
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE , Base.UpdateType.ROLLBACK)
    SELECTION_PREFIX = 'core.tushare'

    @classmethod
    def iter_fetchers(cls , keys : Sequence[str] | None = None) -> Generator[type[TushareFetcher] , None , None]:
        """Iterate over registered tushare fetchers, optionally filtered by class name."""
        TushareFetcher.load_tasks()
        key_set = set(keys) if keys is not None else None
        for name , fetcher in TushareFetcher.registry.items():
            if key_set is not None and name not in key_set:
                continue
            yield fetcher

    @classmethod
    def selection_tree(cls) -> Base.UpdateMenuNode:
        """Menu: task module (t00_info, ...) → fetcher class name."""
        TushareFetcher.load_tasks()
        groups : dict[str , list[type[TushareFetcher]]] = defaultdict(list)
        for fetcher in TushareFetcher.registry.values():
            module_tail = fetcher.__module__.rsplit('.' , 1)[-1]
            groups[module_tail].append(fetcher)
        children : list[Base.UpdateMenuNode] = []
        for module_tail in sorted(groups):
            leaves = [
                Base.UpdateMenuNode(
                    label = fetcher.__name__ ,
                    key = f'{cls.SELECTION_PREFIX}.{fetcher.__name__}' ,
                    help = f'{fetcher.DB_SRC}/{fetcher.DB_KEY} ({fetcher.DB_TYPE})' ,
                )
                for fetcher in sorted(groups[module_tail] , key = lambda f: f.__name__)
            ]
            children.append(Base.UpdateMenuNode(
                label = module_tail ,
                children = leaves ,
                help = f'Tushare fetchers from task/{module_tail}.py' ,
            ))
        return Base.UpdateMenuNode(
            label = 'Tushare' ,
            children = children ,
            help = 'Tushare fetchers grouped by task module' ,
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
        keys = [
            key.rsplit('.' , 1)[-1]
            for key in selection
            if key.startswith(f'{cls.SELECTION_PREFIX}.')
        ]
        if not keys:
            return Base.UpdateFlag.SKIPPED
        force , start , end = Base.resolve_force_range(force , start , end , key = 'tushare')
        if force:
            cls.SetClassVB(kwargs.get('vb_level' , 1) , kwargs.get('indent' , 0))
            return cls.proceed_update(
                update_type = Base.UpdateType.UPDATE ,
                keys = keys ,
                overwrite = True ,
                start = start ,
                end = end ,
            )
        return cls.update(keys = keys , indent = kwargs.get('indent' , 0) , vb_level = kwargs.get('vb_level' , 1))

    @classmethod
    def proceed_update(
        cls , update_type : Base.UpdateType , rollback_date : int | None = None ,
        keys : Sequence[str] | None = None ,
        overwrite : bool = False ,
        start : int | None = None ,
        end : int | None = None ,
        **kwargs
    ) -> Base.UpdateFlag:
        flags = Base.UpdateFlagList()
        rollback_date = rollback_date if update_type == Base.UpdateType.ROLLBACK else None
        use_overwrite = bool(overwrite) and rollback_date is None
        TSBackUpDataTransform.clear(rollback_date = rollback_date)
        for fetcher in cls.iter_fetchers(keys = keys):
            flags += fetcher.update(
                rollback_date = rollback_date ,
                overwrite = use_overwrite ,
                start = start ,
                end = end ,
                indent = cls.indent + 1 ,
                vb_level = cls.vb_level + 1 ,
            )
        TSBackUpDataTransform.update()
        return flags.summarize()
