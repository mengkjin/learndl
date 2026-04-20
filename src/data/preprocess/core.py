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
from typing import Any , Type , Literal

from src.proj import Proj , Logger , CALENDAR , Dates , Duration , Const
from src.data.util import DataBlock
from src.data.loader import BlockLoader , FactorCategory1Loader

TRADE_FEAT : list[str] = ['open','close','high','low','vwap','turn_fl']

class PreProcessorMeta(ABCMeta):
    """
    Metaclass for ``PreProcessor`` that auto-registers concrete subclasses.

    A class is registered when:
    - It has no abstract methods remaining (it is concrete).
    - Its name starts with ``'PrePro_'``.

    The registration key is derived from the class name by stripping the
    ``'PrePro_'`` prefix and lowercasing (via ``PreProcessorProperty('key')``).
    """
    registry : dict[str,Type['PreProcessor'] | Any] = {}
    def __new__(cls , name , bases , dct):
        """Create the class and register it if it satisfies the PrePro_ conditions."""
        new_cls = super().__new__(cls , name , bases , dct)
        abstract_methods = getattr(new_cls , '__abstractmethods__' , None)
        if not abstract_methods and name.startswith('PrePro_'):
            assert name not in cls.registry , f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
            cls.registry[getattr(new_cls , 'key')] = new_cls
        return new_cls

class PreProcessorProperty:
    """
    Read-only class-level descriptor that computes and caches a derived property
    for each ``PreProcessor`` subclass.

    Used to lazily derive ``key``, ``category0``, and ``category1`` from the
    class name or CONST config, caching the result per owner class so that
    repeated accesses are O(1).
    """
    def __init__(self , method : str):
        """Register which compute method to call (``'key'``, ``'category0'``, or ``'category1``)."""
        assert method in dir(self) , f'{method} is not in {dir(self)}'
        self.method = method
        self.cache_values = {}

    def __get__(self , instance , owner) -> str:
        """Return the cached value, computing it on first access per owner class."""
        if owner not in self.cache_values:
            self.cache_values[owner] = getattr(self , self.method)(owner)
        return self.cache_values[owner]

    def __set__(self , instance , value):
        """Prevent assignment — these properties are read-only."""
        raise AttributeError(f'{instance.__class__.__name__}.{self.method} is read-only attributes')

    def key(self , owner) -> str:
        """Derive the registration key: strip ``'PrePro_'`` prefix and lowercase."""
        s = str(owner.__qualname__)
        if not s.startswith('PrePro_'):
            return s
        return s.removeprefix('PrePro_').lower()

    def category0(self , owner) -> str:
        """Look up the category0 for a FactorPreProcessor via CONST config."""
        assert issubclass(owner , FactorPreProcessor) , f'{owner.__class__.__name__} must be a FactorPreProcessor'
        return Const.Factor.STOCK.cat1_to_cat0(owner.category1)

    def category1(self , owner) -> str:
        """Derive category1 by stripping ``'PrePro_'`` prefix from the class name."""
        assert issubclass(owner , FactorPreProcessor) , f'{owner.__class__.__name__} must be a FactorPreProcessor'
        return str(owner.__qualname__).removeprefix('PrePro_').lower()

class PreProcessor(metaclass=PreProcessorMeta):
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
        Offset from today (in trading days, negative) used as the start date in
        ``'predict'`` mode.
    fit_start / hist_start / hist_end : int
        Date range used for ``'fit'`` mode and historical normalisation.
    """
    EXTENSION_OVERLAY  : int = 10
    CALCULATION_WINDOW : int = 1  # can be set slightly larger than the calculation window of the factor
    key = PreProcessorProperty('key')
    
    predict_start = -100
    fit_start     = 20070101
    hist_start    = fit_start
    hist_end      = 20161231

    def __init__(self , type : Literal['fit' , 'predict'] = 'fit' , * , mask : dict[str,Any] | None = None , **kwargs):
        """
        Parameters
        ----------
        type : 'fit' | 'predict'
            Controls the start date and whether dump saving is allowed.
        mask : dict | None
            Masking rules forwarded to ``DataBlock.mask_values``.
            Defaults to ``{'list_dt': 91}`` (blank first 91 days post-IPO).
        """
        self.type : Literal['fit' , 'predict'] = type
        self.mask = mask or {'list_dt': 91}
        self.load_start = self.start_date(type)
        self.load_end   = CALENDAR.updated()

    def __repr__(self):
        """Return the class name as the string representation."""
        return f'{self.__class__.__name__}'

    @property
    def enable_saving(self) -> bool:
        """Whether this instance is allowed to write dump/norm files to disk."""
        if not hasattr(self , '_enable_dump_save'):
            self._enable_dump_save = True
        return self._enable_dump_save

    @enable_saving.setter
    def enable_saving(self , value : bool):
        """Disable dump saving (set automatically when loading for a query rather than a full update)."""
        self._enable_dump_save = value

    @abstractmethod
    def process(self, blocks : dict[str,DataBlock]) -> DataBlock: ...
    @abstractmethod
    def block_loaders(self) -> dict[str,BlockLoader]: ...
    @abstractmethod
    def final_feat(self) -> list[str] | None: ...

    @classmethod
    def start_date(cls , type : Literal['fit' , 'predict'] = 'predict') -> int:
        """Return the absolute start date (yyyyMMdd int) for the given ``type``."""
        return CALENDAR.td(CALENDAR.updated() , cls.predict_start).td if type == 'predict' else cls.fit_start

    def load_blocks(self , start = None , end = None , secid : np.ndarray | None = None , indent = 0 , vb_level : Any = 1 , **kwargs):
        """
        Load raw DataBlocks from all registered ``block_loaders``.

        Aligns each block to (secid, date) before returning.  The secid from the
        first block is used to constrain all subsequent blocks.
        """
        loaders = self.block_loaders()
        
        blocks : dict[str,DataBlock] = {}
        vb_level = Proj.vb(vb_level)
        date = CALENDAR.range(start , end)
        for src_key , loader in loaders.items():
            blocks[src_key] = loader.load(start , end , indent = indent + 1 , vb_level = vb_level + 1 , **kwargs)
            blocks[src_key] = blocks[src_key].align_secid_date(secid , date , inplace = True)
            secid = blocks[src_key].secid
        return blocks
    
    def process_blocks(self, blocks : dict[str,DataBlock]):
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

    def pre_process(self , start : int | None = None , end : int | None = None , * , secid : np.ndarray | None = None , indent = 0 , vb_level : Any = 'max' , **kwargs) -> DataBlock:
        """
        Load raw blocks, apply the transformation, slice to [start, end], and apply masking.

        This is the core compute path.  Called by ``load_with_extension`` for each
        missing date span.  Does not touch the disk.
        """
        if start is None:
            start = self.fit_start if self.type == 'fit' else self.predict_start
        if end is None:
            end = self.load_end
        if start > end:
            return DataBlock()

        with Logger.Timer(f'[{self.key}] blocks loading' , indent = indent , vb_level = vb_level , enter_vb_level = vb_level + 2):
            load_start = CALENDAR.td(start , -self.CALCULATION_WINDOW + 1).td
            block_dict = self.load_blocks(load_start, end, secid = secid, indent = indent + 1 , vb_level = vb_level + 2)

        with Logger.Timer(f'[{self.key}] blocks process' , indent = indent , vb_level = vb_level + 1):
            block = self.process_blocks(block_dict)
            block = block.slice_date(start , end)

        with Logger.Timer(f'[{self.key}] blocks masking' , indent = indent , vb_level = vb_level + 1):   
            block = block.mask_values(mask = self.mask)
            
        return block

    def load(self , dates : np.ndarray | list[int], * , secid : np.ndarray | None = None , indent : int = 0 , vb_level : Any = 'max') -> DataBlock:
        """Load data for the given dates, disabling dump saving (query mode)."""
        return self.load_with_extension(dates_for_query = dates , secid = secid, indent = indent , vb_level = vb_level + 1)

    def load_dump(self , indent : int = 0 , vb_level : Any = 'max'):
        """Load the preprocessed dump from disk; returns empty DataBlock if not found."""
        if not self.dump_exists():
            return DataBlock()
        with Logger.Timer(f'[{self.key}] dumped loading' , indent = indent , vb_level = vb_level):
            block = DataBlock.load_dump(category = 'preprocess' , preprocess_key = self.key , type = self.type)
        return block

    def save_dump(self , block : DataBlock , indent : int = 0 , vb_level : Any = 'max'):
        """Save the block as a preprocessed dump if ``enable_saving`` is True."""
        if not self.enable_saving:
            return
        with Logger.Timer(f'[{self.key}] blocks dumping' , indent = indent , vb_level = vb_level):
            block.set_flags(category = 'preprocess' , preprocess_key = self.key , type = self.type).save_dump()

    def dump_exists(self) -> bool:
        """Return True if a preprocessed dump file already exists on disk."""
        return DataBlock.path_preprocess(self.key , self.type).exists()

    def save_norm(self , block : DataBlock , indent : int = 0 , vb_level : Any = 'max'):
        """Compute and save historical normalisation statistics for this key (fit mode only)."""
        if self.type != 'fit' or not self.enable_saving:
            return
        with Logger.Timer(f'[{self.key}] blocks norming' , indent = indent , vb_level = vb_level):
            block.hist_norm(self.key , self.hist_start , self.hist_end)

    def load_with_extension(self , dates_for_query : np.ndarray | list[int] | None = None, * , secid : np.ndarray | None = None , indent : int = 3 , vb_level : Any = 'max') -> DataBlock:
        """
        load data with extension , try dumped data first , then extend to the end date
        Args:
            dates_for_query: the dates to query the data for , if supplied, this is not an update mission so enable_dump_save is set to False
        Returns:
            tuple[DataBlock , bool]: the data block and a boolean indicating if the data is extended
        """
        vb_level = Proj.vb(vb_level)
        
        block = self.load_dump(indent = indent , vb_level = vb_level + 1).align_secid(secid , inplace = True)
        if dates_for_query is None:
            dates_for_query = [self.load_start , self.load_end]
            start , end = self.load_start , self.load_end
        else:
            start , end = min(dates_for_query) , max(dates_for_query)
            self.enable_saving = False
        block = block.slice_date(start , end)
        if not block.empty:
            block_start , block_end = block.date[0] , block.date[-1]
            if block_end >= end and block_start <= start:
                return block
            
        span_tuples : list[tuple[int,int]] = []
        block_start , block_end = block.first_valid_date , block.last_valid_date
        if block.empty:
            span_tuples.append((start , end))
        elif len(dates_for_query) > 0:
            if start < block_start:
                span_tuples.append((start , CALENDAR.td(block_start , -1).td))
            if end > block_end:
                span_tuples.append((CALENDAR.td(block_end , -self.EXTENSION_OVERLAY + 1).td , end))
        extentions : list[DataBlock] = []
        for span_start , span_end in span_tuples:
            span_load_start = CALENDAR.td(span_start , -self.CALCULATION_WINDOW + 1).td
            ext = self.pre_process(span_load_start , span_end , secid = secid, indent = indent + 1 , vb_level = vb_level + 1).slice_date(span_start , span_end)
            if not ext.empty:
                extentions.append(ext) 
        if not extentions:
            return block 

        with Logger.Timer(f'[{self.key}] blocks merging' , indent = indent , vb_level = vb_level + 2):
            block = block.merge_others(*extentions , inplace = True).slice_date(start , end)

        return block

    def should_be_skipped(self , force_update : bool = False , indent : int = 1 , vb_level : Any = 2):
        """Return True if the dump was already updated today and ``force_update`` is False."""
        modified_time = DataBlock.last_preprocess_time(self.key , self.type)
        if not force_update and CALENDAR.is_updated_today(modified_time):
            time_str = datetime.strptime(str(modified_time) , '%Y%m%d%H%M%S').strftime("%Y-%m-%d %H:%M:%S")
            Logger.skipping(f'[{self.key.upper()}] already preprocessing at {time_str}!' , indent = indent + 1 , vb_level = vb_level + 1)
            return True
        return False

    def update(self , force_update : bool = False , indent : int = 1 , vb_level : Any = 2):
        """
        Run the full incremental update: extend dump, save dump, save norms.

        Skips if the dump was already updated today (unless ``force_update=True``).
        Calls ``load_with_extension(dates_for_query=None)`` which triggers a full
        date-range update from ``load_start`` to ``load_end``.
        """
        vb_level = Proj.vb(vb_level)
        if self.should_be_skipped(force_update , indent + 1 , vb_level + 1):
            return

        tt1 = datetime.now()
        Logger.stdout(f'Update Preprocess [{self.key.upper()}] for {"fitting" if self.type == "fit" else "predicting"} start...' , indent = indent , vb_level = vb_level + 2)
        data_block = self.load_with_extension(dates_for_query = None , indent = indent + 2 , vb_level = vb_level + 2)
        
        self.save_dump(data_block , indent = indent + 2 , vb_level = vb_level + 3)
        self.save_norm(data_block , indent = indent + 2 , vb_level = vb_level + 3)
        
        # gc.collect()
        Logger.success(f'Update Preprocess [{self.key.upper()}] for {"fitting" if self.type == "fit" else "predicting"}  '
                       f'({Dates(data_block.date)}) finished! Cost {Duration(since = tt1)}' , 
                        indent = indent + 1 , vb_level = vb_level + 1)
    
class FactorPreProcessor(PreProcessor):
    """
    Preprocessor that delegates entirely to ``FactorCategory1Loader``.

    Subclasses only need to be named ``PrePro_<category1>``; the category1 is
    derived automatically from the class name.  No ``process`` or ``block_loaders``
    implementation is required.
    """
    category0 = PreProcessorProperty('category0')
    category1 = PreProcessorProperty('category1')

    def block_loaders(self) -> dict[str,BlockLoader]:
        """Return a loader for all factors in ``category1`` using normalised values."""
        return {'factor' : FactorCategory1Loader(self.category1 , normalize = True , fill_method = 'drop' , notice_empty = False)}

    def final_feat(self):
        """No feature filtering — return all factors from the loader."""
        return None

    def process(self , blocks):
        """Pass the factor block through unchanged."""
        return blocks['factor']

class TradePreProcessor(PreProcessor):
    """
    Preprocessor for daily OHLCV data.

    Restricts output features to ``TRADE_FEAT`` (open/close/high/low/vwap/turn_fl).
    Subclasses must implement ``process`` and ``block_loaders``.
    """
    def final_feat(self):
        """Return the standard OHLCV feature list."""
        return TRADE_FEAT

class MicellaneousPreProcessor(PreProcessor):
    """
    Miscellaneous preprocessor for data that does not fit into general workflow.
    """
    def process(self, blocks : dict[str,DataBlock]): return DataBlock()
    def block_loaders(self) -> dict[str,BlockLoader]: return {}
    def final_feat(self) -> list[str] | None: return None

    @abstractmethod
    def pre_process(self , start : int | None = None , end : int | None = None , * , secid : np.ndarray | None = None , indent = 0 , vb_level : Any = 'max' , **kwargs) -> DataBlock:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement pre_process')
