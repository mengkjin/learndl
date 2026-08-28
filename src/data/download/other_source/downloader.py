"""
Other source downloader , including RiceQuant and Baostock to download minute bars
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.proj import Base
from .rcquant import RcquantMinBarDownloader
from .rcquant.initializer import MinDataType
from .baostock_5m import Baostock5minBarDownloader

__all__ = ['OtherSourceDownloader']

class OtherSourceDownloader(Base.SelectiveUpdateSupport):
    SELECTION_PREFIX = 'core.other'

    @classmethod
    def selection_tree(cls) -> Base.UpdateMenuNode:
        rcquant_leaves = [
            Base.UpdateMenuNode(
                label = data_type.value ,
                key = f'{cls.SELECTION_PREFIX}.rcquant.{data_type.value}' ,
                help = f'RiceQuant 1-min bars for {data_type.value}' ,
            )
            for data_type in MinDataType
        ]
        return Base.UpdateMenuNode(
            label = 'OtherSource' ,
            help = 'Minute bars from RiceQuant / Baostock' ,
            children = [
                Base.UpdateMenuNode(
                    label = 'Rcquant' ,
                    children = rcquant_leaves ,
                    help = 'RiceQuant minute bars by instrument type' ,
                ) ,
                Base.UpdateMenuNode(
                    label = 'Baostock' ,
                    key = f'{cls.SELECTION_PREFIX}.baostock' ,
                    help = 'Temporarily disabled: baostock download hangs with no socket timeout' ,
                    disabled = not Baostock5minBarDownloader.ENABLED ,
                ) ,
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
        rcquant_types = [
            key.rsplit('.' , 1)[-1]
            for key in selection
            if key.startswith(f'{cls.SELECTION_PREFIX}.rcquant.')
        ]
        do_baostock = (
            Baostock5minBarDownloader.ENABLED
            and f'{cls.SELECTION_PREFIX}.baostock' in selection
        )
        if not rcquant_types and not do_baostock:
            return Base.UpdateFlag.SKIPPED

        flags = Base.UpdateFlagList()
        if rcquant_types:
            force_rq , start_rq , end_rq = Base.resolve_force_range(
                force , start , end , key = 'rcquant_min')
            if force_rq:
                assert start_rq is not None and end_rq is not None
                RcquantMinBarDownloader.SetClassVB(
                    kwargs.get('vb_level' , 1) , kwargs.get('indent' , 0))
                flags += RcquantMinBarDownloader.proceed_update(
                    start = start_rq , end = end_rq ,
                    data_types = rcquant_types , overwrite = True ,
                )
            else:
                flags += RcquantMinBarDownloader.update(
                    data_types = rcquant_types ,
                    indent = kwargs.get('indent' , 0) ,
                    vb_level = kwargs.get('vb_level' , 1) ,
                )
        if do_baostock:
            force_bs , start_bs , end_bs = Base.resolve_force_range(
                force , start , end , key = 'baostock_5min')
            if force_bs:
                assert start_bs is not None and end_bs is not None
                Baostock5minBarDownloader.SetClassVB(
                    kwargs.get('vb_level' , 1) , kwargs.get('indent' , 0))
                flags += Baostock5minBarDownloader.proceed_update(
                    start = start_bs , end = end_bs , overwrite = True ,
                )
            else:
                flags += Baostock5minBarDownloader.update(
                    indent = kwargs.get('indent' , 0) ,
                    vb_level = kwargs.get('vb_level' , 1) ,
                )
        return flags.summarize()

    @classmethod
    def update(cls , * , indent: int = 0, vb_level: int = 1) -> Base.UpdateFlagList:
        flags = Base.UpdateFlagList()
        flags += RcquantMinBarDownloader.update(indent=indent, vb_level=vb_level)
        if Baostock5minBarDownloader.ENABLED:
            flags += Baostock5minBarDownloader.update(indent=indent, vb_level=vb_level)
        else:
            Baostock5minBarDownloader.logger.skipping(
                'Baostock 5min download temporarily disabled (no socket timeout)')
            flags += Base.UpdateFlag.SKIPPED
        return flags
