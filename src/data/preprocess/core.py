"""
Preprocessing pipeline base classes for the data pipeline.

``PreProcessor`` subclasses (named ``PrePro_<key>``) are auto-registered in
``PreProcessorMeta.registry`` via the metaclass.  Each subclass transforms raw
database blocks into a normalised ``DataBlock`` for model training or prediction.

Hierarchy
---------
PreProcessor               — abstract base; handles incremental extension/caching
  FactorPreProcessor       — delegates to FactorCategory1Loader (auto factor loading)
  TradePreProcessor        — OHLCV-specific; sets final_feat to TRADE_FEAT
  MicellaneousPreProcessor — fully custom; overrides pre_process entirely
"""
from __future__ import annotations

import numpy as np

from abc import abstractmethod , ABCMeta
from datetime import datetime
from functools import cached_property
from typing import Any , cast , get_args

from src.proj import CALENDAR , Const , Base , Dates
from src.data.util import DataBlock
from src.data.loader import BlockLoader , FactorCategory1Loader

__all__ = ['PreProcessor' , 'FactorPreProcessor' , 'TradePreProcessor' , 'MicellaneousPreProcessor']

TRADE_FEAT : tuple[str,...] = ('open','close','high','low','vwap','turn_fl')

class PreProcessorMeta(ABCMeta):
    """
    Metaclass for ``PreProcessor`` that auto-registers concrete subclasses.

    A class is registered when:
    - It has no abstract methods remaining (it is concrete).
    - Its name starts with ``'PrePro_'``.

    The registration key is derived from the class name by stripping the
    ``'PrePro_'`` prefix and lowercasing (via ``PreProcessorProperty('key')``).
    """
    registry : dict[str,type[PreProcessor] | Any] = {}
    def __new__(cls , name , bases , dct):
        """Create the class and register it if it satisfies the PrePro_ conditions."""
        new_cls = super().__new__(cls , name , bases , dct)
        abstract_methods = getattr(new_cls , '__abstractmethods__' , None)
        if not abstract_methods and name.startswith('PrePro_'):
            assert name not in cls.registry , f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
            cls.registry[getattr(new_cls , 'key')] = new_cls
        return new_cls

class _PPKey:
    """
    return the key of the preprocessor
    """
    def __init__(self):
        """Register which compute method to call (``'key'``, ``'category0'``, or ``'category1``)."""
        self.cache_values = {}

    def __get__(self , instance , owner : type[PreProcessor]) -> str:
        """Return the cached value, computing it on first access per owner class."""
        if owner not in self.cache_values:
            self.cache_values[owner] = str(owner.__qualname__).removeprefix('PrePro_').lower()
        return self.cache_values[owner]

class _FactorPPCategory0:
    """
    return the category0 of the preprocessor
    """
    def __init__(self):
        """Register which compute method to call (``'key'``, ``'category0'``, or ``'category1``)."""
        self.cache_values = {}

    def __get__(self , instance , owner : type[FactorPreProcessor]) -> Base.lit.FactorCategory0:
        """Return the cached value, computing it on first access per owner class."""
        if owner not in self.cache_values:
            self.cache_values[owner] = Const.Factor.STOCK.cat1_to_cat0(cast(Base.lit.FactorCategory1, owner.category1))
        return self.cache_values[owner]

class _FactorPPCategory1:
    """
    return the category1 of the preprocessor
    """
    def __init__(self):
        """Register which compute method to call (``'key'``, ``'category0'``, or ``'category1``)."""
        self.cache_values = {}

    def __get__(self , instance , owner : type[FactorPreProcessor]) -> Base.lit.FactorCategory1:
        """Return the cached value, computing it on first access per owner class."""
        if owner not in self.cache_values:
            cat1 = owner.key
            assert cat1 in get_args(Base.lit.FactorCategory1) , f'{cat1} is not in {Base.lit.FactorCategory1}'
            self.cache_values[owner] = cast(Base.lit.FactorCategory1, cat1)
        return self.cache_values[owner]
class PreProcessor(Base.BoundLogger, metaclass=PreProcessorMeta):
    """
    Abstract base class for all data preprocessors.

    Subclasses must implement:
    - ``process(blocks)``         — transforms raw DataBlocks into the output DataBlock.
    - ``block_loaders()``         — returns a dict of ``BlockLoader`` instances.
    - ``final_feat()``            — returns the feature list to retain (or None to keep all).

    Concrete subclasses named ``PrePro_<key>`` are auto-registered at class creation.

    Class Attributes
    ----------------
    EXTENSION_OVERLAY : int
        Number of trading days to overlap when extending an existing dump to prevent
        edge-discontinuities from rolling-window calculations.
    CALCULATION_WINDOW : int
        Extra trading days to load *before* the requested start date so that
        rolling calculations have sufficient history.
    predict_start : int
        Offset from update_to (in trading days, negative) used as the start date in
        ``'predict'`` mode.
    fit_start / hist_start / hist_end : int
        Date range used for ``'fit'`` mode and historical normalisation.
    """
    EXTENSION_OVERLAY  : int = 10
    CALCULATION_WINDOW : int = 1  # can be set slightly larger than the calculation window of the factor
    key = _PPKey()
    
    predict_start = -100
    fit_start     = 20070101
    hist_start    = fit_start
    hist_end      = 20161231

    def __init__(
        self , frame : Base.lit.DataBlockTimeFrame = 'fit' , * , 
        mask : dict[str, Any] | None = None , 
        indent : int = 0 , vb_level : Base.lit.VerbosityLevel = 'max' , **kwargs
    ) -> None:
        """
        Parameters
        ----------
        frame : 'fit' or 'predict'
            Controls the start date and whether dump saving is allowed.
        mask : dict[str,Any] | None
            Masking rules forwarded to ``DataBlock.mask_values``.
            Defaults to ``{'list_dt': 91}`` (blank first 91 days post-IPO).
        """
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.frame : Base.lit.DataBlockTimeFrame = frame
        self.mask = mask or {'list_dt': 91}
        self.load_start = self.start_date(frame)
        self.load_end   = CALENDAR.updated()

    def __repr__(self):
        """Return the class name as the string representation."""
        return f'{self.__class__.__name__}'

    @cached_property
    def enable_saving(self) -> bool:
        """Whether this instance is allowed to write dump/norm files to disk."""
        return True

    @abstractmethod
    def process(self, blocks : dict[str,DataBlock]) -> DataBlock: 
        ...
    @abstractmethod
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]: 
        ...
    @abstractmethod
    def final_feat(self) -> list[str] | None: 
        ...

    @classmethod
    def start_date(cls , frame : Base.lit.DataBlockTimeFrame = 'predict') -> int:
        """Return the absolute start date (yyyyMMdd int) for the given ``frame``."""
        return CALENDAR.td(CALENDAR.updated() , cls.predict_start).td if frame == 'predict' else cls.fit_start

    def load_blocks(self , start = None , end = None , secid : Base.alias.SecidType = None , **kwargs) -> dict[str,DataBlock]:
        """
        Load raw DataBlocks from all registered ``block_loaders``.

        Aligns each block to (secid, date) before returning.  The secid from the
        first block is used to constrain all subsequent blocks.
        """
        loaders = self.block_loaders(indent = self.indent + 1 , vb_level = self.vb_level + 3)
        
        blocks : dict[str,DataBlock] = {}
        date = CALENDAR.range(start , end)
        for src_key , loader in loaders.items():
            blocks[src_key] = loader.load(start , end , **kwargs)
            blocks[src_key] = blocks[src_key].align_secid_date(secid , date , inplace = True)
            secid = blocks[src_key].secid
        return blocks
    
    def process_blocks(self, blocks : dict[str,DataBlock]) -> DataBlock:
        """
        Call ``process(blocks)`` with numpy error suppression and apply final feature alignment.

        Returns an empty DataBlock if any input block is empty.
        """
        if any([block.empty for block in blocks.values()]):
            return DataBlock()
        np.seterr(invalid = 'ignore' , divide = 'ignore')
        data_block = self.process(blocks)
        data_block = data_block.align_feature(self.final_feat() , inplace = True)
        np.seterr(invalid = 'warn' , divide = 'warn')
        return data_block

    def pre_process(
        self , start : int | None = None , end : int | None = None , * , 
        secid : Base.alias.SecidType = None , **kwargs
    ) -> DataBlock:
        """
        Load raw blocks, apply the transformation, slice to [start, end], and apply masking.

        This is the core compute path.  Called by ``load_with_extension`` for each
        missing date span.  Does not touch the disk.
        """
        if start is None:
            start = self.fit_start if self.frame == 'fit' else self.predict_start
        if end is None:
            end = self.load_end
        if start > end:
            return DataBlock()

        with self.logger.timer(f'{self.key} blocks loading' , vb = 3 , enter_vb = 4):
            load_start = CALENDAR.td(start , -self.CALCULATION_WINDOW + 1).td
            block_dict = self.load_blocks(load_start, end, secid = secid)

        with self.logger.timer(f'{self.key} blocks process' , vb = 3):
            block = self.process_blocks(block_dict)
            block = block.slice_date(start , end)

        with self.logger.timer(f'{self.key} blocks masking' , vb = 3):   
            block = block.mask_values(mask = self.mask)
            
        return block

    def load(self , dates : Base.alias.DateType, * , secid : Base.alias.SecidType = None) -> DataBlock:
        """Load data for the given dates, disabling dump saving (query mode)."""
        return self.load_with_extension(dates_for_query = dates , secid = secid)

    def load_dump(self , reconstruct : bool = False , rollback_date : int | None = None) -> DataBlock:
        """Load the preprocessed dump from disk; returns empty DataBlock if not found."""
        if not self.dump_exists() or reconstruct:
            return DataBlock()
        with self.logger.timer(f'{self.key} dumped loading' , vb = 2):
            block = DataBlock.load_dump(category = 'preprocess' , preprocess_key = self.key , frame = self.frame)
            if rollback_date:
                block = block.slice_date(None , rollback_date - 1)
        return block

    def save_dump(self , block : DataBlock) -> None:
        """Save the block as a preprocessed dump if ``enable_saving`` is True."""
        if not self.enable_saving:
            return
        with self.logger.timer(f'{self.key} blocks dumping' , vb = 3):
            block.set_flags(category = 'preprocess' , preprocess_key = self.key , frame = self.frame).save_dump()

    def dump_exists(self) -> bool:
        """Return True if a preprocessed dump file already exists on disk."""
        return DataBlock.path_preprocess(self.key , self.frame).exists()

    def save_norm(self , block : DataBlock) -> None:
        """Compute and save historical normalisation statistics for this key (fit mode only)."""
        if self.frame != 'fit' or not self.enable_saving:
            return
        with self.logger.timer(f'{self.key} blocks norming' , vb = 3):
            block.hist_norm(self.key , self.hist_start , self.hist_end)

    def load_with_extension(
        self , dates_for_query : Base.alias.DateType = None, * , 
        secid : Base.alias.SecidType = None , reconstruct : bool = False , rollback_date : int | None = None
    ) -> DataBlock:
        """
        load data with extension , try dumped data first , then extend to the end date
        Args:
            dates_for_query: the dates to query the data for , if supplied, this is not an update mission so enable_dump_save is set to False
        Returns:
            tuple[DataBlock , bool]: the data block and a boolean indicating if the data is extended
        """
        block = self.load_dump(reconstruct = reconstruct , rollback_date = rollback_date)
        
        block = block.align_secid(secid , inplace = True)

        dates_for_query = Dates(dates_for_query)
        if dates_for_query.empty:
            start , end = self.load_start , self.load_end
        else:
            start , end = dates_for_query.min , dates_for_query.max
            self.enable_saving = False
        block = block.slice_date(start , end)

        if not block.empty:
            block_start , block_end = block.date[0] , block.date[-1]
            if block_end >= end and block_start <= start:
                return block
            
        span_tuples : list[tuple[int,int]] = []
        meaningful_dates = block.meaningful_dates
        if len(meaningful_dates) == 0:
            block_start , block_end = 99991231 , 19000101
        else:
            block_start , block_end = meaningful_dates[0] , meaningful_dates[-1]

        if block.empty:
            span_tuples.append((start , end))
        elif dates_for_query:
            if start < block_start:
                span_tuples.append((start , CALENDAR.td(block_start , -1).td))
            if end > block_end:
                span_tuples.append((CALENDAR.td(block_end , -self.EXTENSION_OVERLAY + 1).td , end))

        extentions : list[DataBlock] = []
        for span_start , span_end in span_tuples:
            span_load_start = CALENDAR.td(span_start , -self.CALCULATION_WINDOW + 1).td
            ext = self.pre_process(span_load_start , span_end , secid = secid).slice_date(span_start , span_end)
            if not ext.empty:
                extentions.append(ext) 
        if not extentions:
            return block 

        with self.logger.timer(f'{self.key} blocks merging' , vb = 2):
            block = block.merge_others(*extentions , inplace = True).slice_date(start , end)

        return block

    def should_be_skipped(self , force_update : bool = False) -> bool:
        """Return True if the dump was already updated and ``force_update`` is False."""
        modified_time = DataBlock.last_preprocess_time(self.key , self.frame)
        if not force_update and CALENDAR.is_updated_today(modified_time):
            time_str = datetime.strptime(str(modified_time) , '%Y%m%d%H%M%S').strftime("%Y-%m-%d %H:%M:%S")
            self.logger.skipping(f'[{self.key.upper()}] already preprocessing at {time_str}!' , add_prefix = False)
            return True
        return False

    def build(
        self , * ,  
        force_build : bool = False , 
        reconstruct : bool = False , 
        rollback_date : int | None = None ,
        confirm : bool = True
    ) -> Base.UpdateFlag:
        """
        Run the building process: extend dump, save dump, save norms.

        Skips if the dump was already updated (unless ``force_build=True``).
        Calls ``load_with_extension(dates_for_query=None)`` which triggers a full
        date-range building from ``load_start`` to ``load_end``.
        """
        if reconstruct:
            self.logger.critical('Reconstructing the preprocessed data...')
            if confirm:
                from src.proj.util.functional.ask import AskFor
                flag = AskFor.Confirmation(title = 'Are you sure to reconstruct the preprocessed data?')
                if not flag.valid:
                    return Base.UpdateFlag.FAILED
            force_build = True

        if self.should_be_skipped(force_build):
            return Base.UpdateFlag.SKIPPED

        tt1 = datetime.now()
        if reconstruct:
            status = 'reconstruct'
        elif rollback_date:
            CALENDAR.check_rollback_date(rollback_date)
            status = f'rollback from {rollback_date}'
        else:
            status = 'update'
        self.logger.stdout(f'{status.upper()} Preprocessed ({self.frame}) of [{self.key.upper()}] start...' , vb = 2 , add_prefix = False)
        data_block = self.load_with_extension(dates_for_query = None , reconstruct = reconstruct , rollback_date = rollback_date)
        
        self.save_dump(data_block)
        self.save_norm(data_block)
        
        # gc.collect()
        self.logger.success(
            f'{status.upper()} Preprocessed ({self.frame}) of [{self.key.upper()}] at '
            f'{Dates(data_block.date)} finished! Cost {Base.Since(tt1)}' , 
            add_prefix = False)
        return Base.UpdateFlag.SUCCESS
    
class FactorPreProcessor(PreProcessor):
    """
    Preprocessor that delegates entirely to ``FactorCategory1Loader``.

    Subclasses only need to be named ``PrePro_<category1>``; the category1 is
    derived automatically from the class name.  No ``process`` or ``block_loaders``
    implementation is required.
    """
    category0 = _FactorPPCategory0()
    category1 = _FactorPPCategory1()

    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        """Return a loader for all factors in ``category1`` using normalised values."""
        return {
            'factor' : FactorCategory1Loader(self.category1 , normalize = True , fill_method = 'drop' , notice_empty = False, **kwargs)}

    def final_feat(self) -> list[str] | None:
        """No feature filtering — return all factors from the loader."""
        return None

    def process(self , blocks : dict[str,DataBlock]) -> DataBlock:
        """Pass the factor block through unchanged."""
        return blocks['factor']

class TradePreProcessor(PreProcessor):
    """
    Preprocessor for daily OHLCV data.

    Restricts output features to ``TRADE_FEAT`` (open/close/high/low/vwap/turn_fl).
    Subclasses must implement ``process`` and ``block_loaders``.
    """
    def final_feat(self) -> list[str] | None:
        """Return the standard OHLCV feature list."""
        return [*TRADE_FEAT]

class MicellaneousPreProcessor(PreProcessor):
    """
    Miscellaneous preprocessor for data that does not fit into general workflow.
    """
    def process(self, blocks : dict[str,DataBlock]): 
        return DataBlock()
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]: 
        return {}
    def final_feat(self) -> list[str] | None: 
        return None

    @abstractmethod
    def pre_process(
        self , start : int | None = None , end : int | None = None , * , 
        secid : Base.alias.SecidType = None , **kwargs
    ) -> DataBlock:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement pre_process')
