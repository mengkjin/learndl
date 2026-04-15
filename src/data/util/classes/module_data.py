"""
Multi-block data loader and cache manager for model training and prediction.

``ModuleData`` orchestrates loading a set of named ``DataBlock`` objects
(``'y'``, ``'day'``, ``'15m'``, custom factor blocks, etc.) for a given
training or prediction session.  It handles:

- Incremental extension: only missing dates are loaded from preprocessed dumps.
- Alignment: all blocks are brought to a common (secid, date) grid.
- Caching: fully-aligned blocks are saved to disk via ``DataCache`` and
  reloaded on subsequent calls (only if the cache is up-to-date).
- Normalisation: ``DataBlockNorm`` objects are loaded alongside the data.
- Filtering: optional ``SecidFilter`` and ``DateFilter`` can restrict the
  universe and date range during training.

Helper classes
--------------
SecidFilter
    Restricts the security universe (random sample, head-N, or benchmark).
DateFilter
    Restricts the date range via a ``'yyyyMMdd~yyyyMMdd'`` string.
"""
import torch
import numpy as np

from copy import deepcopy
from functools import partial
from typing import Any , Literal

from src.proj import Logger , Proj , CALENDAR , Dates
from src.proj.util import properties

from .data_block import DataBlock , DataBlockNorm , data_type_abbr
from .datacache import DataCache
from .special_dataset import SpecialDataSet

__all__ = ['ModuleData']

class ModuleData:
    """
    Orchestrator for multi-block model input data.

    Parameters
    ----------
    data_type_list : list[str]
        Block keys to load in addition to ``'y'`` (e.g. ``['day', '15m']``).
    y_labels : list[str] | None
        Sub-set of features to expose via the ``y`` property.
    use_data : 'fit' | 'predict' | 'both'
        Controls the date range and whether the disk cache is used.
    factor_names : list[str] | None
        Optional list of named factors to load via ``FactorLoader``.
    factor_start_dt / factor_end_dt : int | None
        Date range to apply when loading factor data.
    filter_secid : str | None
        SecidFilter specification string (e.g. ``'random.200'``, ``'csi300'``).
    filter_date : str | None
        DateFilter specification string (e.g. ``'20100101~20201231'``).
    indent / vb_level : int
        Verbosity / indentation for the Logger.Timer messages.
    dtype : torch dtype
        Dtype to cast loaded blocks to (default: ``torch.float``).
    """
    def __init__(
        self , data_type_list : list[str] , y_labels : list[str] | None = None , use_data : Literal['fit' , 'predict' , 'both'] = 'fit' , * ,
        factor_names : list[str] | None = None , factor_start_dt : int | None = None , factor_end_dt : int | None = None , 
        filter_secid : str | None = None , filter_date : str | None = None ,
        indent : int = 1 , vb_level : Any = 2 , dtype = torch.float ,**kwargs
    ):
        self.data_type_list = sorted([self.abbr(data_type) for data_type in data_type_list])
        self.y_labels = y_labels
        self.use_data = use_data
        self.factor_names = factor_names
        self.factor_start_dt = factor_start_dt
        self.factor_end_dt = factor_end_dt

        self.datacache = DataCache(type = 'module_data' , data_type_list = self.data_type_list)
        
        self.indent = indent
        self.vb_level = Proj.vb(vb_level)

        if dtype is None: 
            dtype = torch.float
        if isinstance(dtype , str): 
            dtype = getattr(torch , dtype)
        self.dtype = dtype

        self.secid_filter = SecidFilter(filter_secid if self.use_data == 'fit' else None)
        self.date_filter = DateFilter(filter_date if self.use_data == 'fit' else None)
        self.enable_cache_save = self.enable_cache and filter_secid is None and filter_date is None

        self.kwargs = kwargs

        self.blocks : dict[str,DataBlock] = {}
        self.norms : dict[str,DataBlockNorm] = {}

    @property
    def PrePros(self):
        """Lazy accessor for the ``PrePros`` registry (avoids circular imports at module load)."""
        if not hasattr(self , '_prepros'):
            from src.data.preprocess import PrePros
            self._prepros = PrePros
        return self._prepros

    @property
    def load_keys(self):
        """All block keys to be loaded: ``['y', *data_type_list]``."""
        return ['y' , *self.data_type_list]

    @property
    def x(self) -> dict[str,DataBlock]:
        """All blocks except ``'y'`` (the feature/input blocks)."""
        return {key:value for key,value in self.blocks.items() if key != 'y'}

    @property
    def y(self) -> DataBlock:
        """The label block, feature-aligned to ``y_labels``."""
        return self.blocks['y'].align_feature(self.y_labels)

    @property
    def empty_x(self):
        """True if no feature blocks are loaded or all are empty."""
        return len(self.x) == 0 or all([x.empty for x in self.x.values()])

    @property
    def shape(self):
        """Shape summary across x, y, secid, and date."""
        return properties.shape(self , ['x' , 'y' , 'secid' , 'date'])

    @property
    def secid(self):
        """Security universe from the ``'y'`` block."""
        return self.blocks['y'].secid

    @property
    def date(self):
        """Date range from the ``'y'`` block."""
        return self.blocks['y'].date

    @property
    def enable_cache(self):
        """True when disk caching is active (``use_data`` is ``'fit'``/``'both'`` and cache key exists)."""
        return self.use_data in ['fit' , 'both'] and self.datacache

    @property
    def loaded(self):
        """True after ``load()`` has been called successfully."""
        if not hasattr(self , '_loaded'):
            self._loaded = False
        return self._loaded

    @property
    def block_title(self):
        """Human-readable summary of the block set for log messages."""
        return f'{len(self.load_keys)} DataBlocks' if len(self.load_keys) > 4 else f'DataBlock [{",".join(self.load_keys)}]'

    def __bool__(self):
        """True when at least one non-empty feature block is loaded."""
        return not self.empty_x

    def copy(self):
        """Return a deep copy of this ModuleData instance."""
        return deepcopy(self)

    def date_within(self , start : int , end : int , interval = 1) -> np.ndarray:
        """Return the loaded date array sliced to [start, end] with optional stride."""
        return CALENDAR.slice(self.date , start , end)[::interval]

    def target_start_end(self):
        """
        Compute the (start, end) date range to load data for.

        - ``'predict'`` mode: loads the last 366 calendar days up to today.
        - ``'fit'`` mode: loads from 20070101 up to the last available ``'y'`` dump date.
        """
        start = CALENDAR.td(CALENDAR.updated() , -366).td if self.use_data == 'predict' else 20070101
        end = DataBlock.last_data_date('y' , 'fit') if self.use_data == 'fit' else CALENDAR.updated()
        end = end or CALENDAR.updated()
        return start , end

    def load(self):
        '''
        load all relevant data of this module data, should be called before any other operations
        blocks: ['y' , *data_type_list] DataBlocks
        norms: ['y' , *data_type_list] DataBlockNorms (if exists)
        factor: factor_names DataBlock
        '''
        self.load_cache()
        self.extend_blocks()
        self.align_blocks()
        self.load_norms()
        self.save_cache()
        self.load_factor()
        DataBlock.blocks_ffill(self.blocks , exclude = ['y'])
        self._loaded = True
        return self

    def load_cache(self):
        if not self.enable_cache:
            return
        data , _ = self.datacache.load_data(self.vb_level)
        if data is not None:
            self.blocks , self.norms = data['blocks'] , data['norms']
            Logger.success(f'Loaded DataBlocks from cache {self.datacache.key} of {Dates(self.date)}' , vb_level = self.vb_level + 2)

    def extend_blocks(self):
        """
        Extend all blocks with missing dates from preprocessed dumps.

        Iterates over ``load_keys`` (``'y'`` first, then X blocks).
        The ``'y'`` block drives the secid and date grids that all X blocks
        are aligned to in the subsequent ``align_blocks()`` call.
        """
        start , end = self.date_filter.filter_start_end(*self.target_start_end())
        date = CALENDAR.range(start , end)
        secid = None
        with Logger.Timer(f'Load {self.block_title} at {start}~{end}' , indent = self.indent , vb_level = self.vb_level + 1):
            for i , key in enumerate(self.load_keys):
                current_block = self.blocks.get(key , DataBlock())
                current_dates = current_block.valid_dates
                ext_dates = CALENDAR.diffs(date , current_dates)
                ext_block = self.load_one(key, dates = ext_dates , secid = secid)
                self.blocks[key] = DataBlock.merge([current_block , ext_block] , inplace = True)
                if i == 0:
                    assert key == 'y' , f'y must be the first key'
                    secid = self.secid_filter(self.blocks[key].secid) # use the y_secid to align all other blocks in next step
                    date = self.date_filter(self.blocks[key].date) # use the y_date to align all other blocks in next step
        return self

    def align_blocks(self):
        """Align all loaded blocks to a common (secid, date) grid via ``DataBlock.blocks_align``."""
        if len(self.blocks) <= 1:
            return self
        with Logger.Timer(f'Align {self.block_title}' , indent = self.indent , vb_level = self.vb_level + 1):
            DataBlock.blocks_align(self.blocks , vb_level = self.vb_level + 2)
        index_lens = [block.shape[:2] for block in self.blocks.values()]
        if index_lens:
            assert all([lens == index_lens[0] for lens in index_lens]) , f'{[(name,block.shape) for name,block in self.blocks.items()]}'
        return self

    def load_one(self , key : str , * , dates : np.ndarray , secid : np.ndarray | None = None , **kwargs):
        """
        Load a single block for the given key and date array.

        Dispatches to ``load_preprocess_block`` for registered preprocessors
        or ``load_special_block`` for special dataset candidates.
        Returns an empty ``DataBlock`` when ``dates`` is empty.
        """
        if len(dates) == 0:
            return DataBlock()
        if key in self.PrePros.keys():
            return self.load_preprocess_block(key, dates = dates, secid = secid, vb_level = self.vb_level + 2 , **kwargs)
        elif key in SpecialDataSet.candidates:
            return self.load_special_block(key, dates = dates, secid = secid, vb_level = self.vb_level + 2 , **kwargs)
        else:
            raise ValueError(f'key [{key}] is not supported')

    def load_preprocess_block(self , key : str , * , dates : np.ndarray , secid : np.ndarray | None = None , **kwargs):
        """Load a block via the registered ``PreProcessor`` for ``key``."""
        type = 'predict' if self.use_data == 'predict' else 'fit'
        block = self.PrePros.get_processor(key , type = type).load(dates = dates , secid = secid, indent = self.indent + 1 , vb_level = self.vb_level + 2)
        return block

    def load_special_block(self , key : str , * , dates : np.ndarray , secid : np.ndarray | None = None , **kwargs):
        """Load a block via ``SpecialDataSet`` for non-standard dataset keys."""
        block = SpecialDataSet.load(key, dates = dates , secid = secid, dtype = self.dtype , vb_level = self.vb_level + 2)
        return block

    def load_norms(self):
        """Load ``DataBlockNorm`` objects for all X block types from disk (no-op if already loaded)."""
        if self.norms:
            return
        self.norms.update(DataBlock.load_preprocess_norms(self.data_type_list , dtype = self.dtype))

    def load_factor(self):
        '''load factor data'''
        if not self.factor_names:
            return self
        factor_title = f'{len(self.factor_names)} Factors' if len(self.factor_names) > 1 else f'Factor [{self.factor_names[0]}]'
        start = max(self.factor_start_dt or self.date[0] , self.date[0])
        end = min(self.factor_end_dt or self.date[-1] , self.date[-1])
        with Logger.Timer(f'Load {factor_title} ({start} - {end})' , indent = self.indent , vb_level = self.vb_level + 2):
            from src.data.loader import FactorLoader
            self.blocks['factor'] = FactorLoader(self.factor_names).load(start , end , vb_level = 'never').align_secid_date(self.secid , self.date , inplace = True)
        return self

    def save_cache(self):
        """
        Persist aligned blocks and norms to ``DataCache`` if the valid date range has grown.

        Only saves when ``enable_cache_save`` is True (no secid/date filter active)
        and the new valid date range is wider than what was previously cached.
        """
        if not self.enable_cache_save:
            return
        valid_end   = min(block.last_valid_date for block in self.blocks.values())
        valid_start = max(block.first_valid_date for block in self.blocks.values())
        old_metadata = self.datacache.load_metadata()
        old_valid_end   : int = old_metadata.get('valid_end'   , 19000101)
        old_valid_start : int = old_metadata.get('valid_start' , 99991231)
        if len(CALENDAR.range(old_valid_start , old_valid_end , 'td')) < len(CALENDAR.range(valid_start , valid_end , 'td')):
            metadata = {'valid_end' : int(valid_end) , 'valid_start' : int(valid_start)}
            blocks = {key:value for key,value in self.blocks.items() if key != 'factor'}
            self.datacache.save_data({'blocks' : blocks , 'norms' : self.norms} , vb_level = self.vb_level + 2 , **metadata)
            Logger.success(f'Saved DataBlocks to cache {self.datacache.key}' , vb_level = self.vb_level + 2)

    @staticmethod
    def abbr(data_type : str):
        """Normalise a data-type key via ``data_type_abbr``."""
        return data_type_abbr(data_type)

    def filter_dates(self , start : int | None = None , end : int | None = None , inplace = False):
        """Restrict all blocks to dates in [start, end]; returns self (optionally a copy)."""
        if start is None and end is None:
            return self
        if not inplace:
            self = self.copy()
        date = CALENDAR.slice(self.date , start , end)
        for block in self.blocks.values():
            block = block.align_date(date , inplace = True)
        return self

    def filter_secid(self , secid : np.ndarray | Any | None = None , exclude = False , inplace = False):
        """
        Keep (or exclude) specific secids from all blocks.

        Parameters
        ----------
        secid : array-like | None
            secids to keep or exclude.
        exclude : bool
            If True, remove the listed secids instead of keeping them.
        inplace : bool
            Modify in-place; otherwise return a copy.
        """
        if secid is None:
            return self
        if not inplace:
            self = self.copy()
        mask = np.isin(self.secid , secid)
        secid = self.secid[~mask] if exclude else self.secid[mask]
        for block in self.blocks.values():
            block = block.align_secid(secid , inplace = True)
        return self

class SecidFilter:
    """
    Callable filter that subsets a security universe array.

    Supported ``value`` strings
    ---------------------------
    None
        No filtering; pass-through.
    ``'random.N'``
        Draw N secids uniformly at random.
    ``'first.N'``
        Keep the first N secids (deterministic, by ascending secid order).
    ``'csi300'`` / ``'csi500'`` / ``'csi1000'``
        Return the benchmark constituent list as of a fixed reference date (20200104).
    """
    def __init__(self , value : str | None):
        if value is None:
            self.filter = self.none
        else:
            Logger.alert1(f'filtering secid for ModuleData: {value}')
            if value.startswith('random.'):
                self.filter = partial(self.random , num = int(value.split('.')[1]))
            elif value.startswith('first.'):
                self.filter = partial(self.first , num = int(value.split('.')[1]))
            elif value in ['csi300' , 'csi500' , 'csi1000']:
                self.filter = partial(self.benchmark , bm = value)
            else:
                raise ValueError(f'input.filter.secid {value} is not valid , should be random.200 , first.200 , csi300 , csi500 , csi1000')
        
    def __call__(self , secid : np.ndarray) -> np.ndarray:
        """Apply the configured filter to ``secid`` and return the filtered array."""
        return self.filter(secid)

    def filter_blocks(self , blocks : dict[str,DataBlock]) -> dict[str,DataBlock]:
        """Apply the filter to all blocks in the dict (aligns secid axis of each block)."""
        if not blocks:
            return blocks
        secid = self.filter(blocks['y'].secid)
        for key,block in blocks.items():
            blocks[key] = block.align_secid(secid , inplace = True)
        return blocks

    @staticmethod
    def none(secid : np.ndarray) -> np.ndarray:
        """Pass-through filter; returns ``secid`` unchanged."""
        return secid

    @staticmethod
    def random(secid : np.ndarray , num : int) -> np.ndarray:
        """Return a random sample of ``num`` secids without replacement."""
        return np.random.choice(secid , num , replace = False)

    @staticmethod
    def first(secid : np.ndarray , num : int) -> np.ndarray:
        """Return the first ``num`` secids in the input array."""
        return secid[:num]

    @classmethod
    def Benchmark(cls):
        """Lazy accessor for the ``Benchmark`` class (avoids circular import)."""
        if not hasattr(cls , '_benchmark'):
            from src.res.factor.util.classes.benchmark import Benchmark
            cls._benchmark = Benchmark
        return cls._benchmark

    @classmethod
    def benchmark(cls , secid : np.ndarray , bm : str , date : int = 20200104) -> np.ndarray:
        """Return the constituent secids of benchmark ``bm`` as of ``date``."""
        return cls.Benchmark()(bm).get(date,True).secid

class DateFilter:
    """
    Callable filter that subsets a date array.

    Parses a ``'yyyyMMdd~yyyyMMdd'`` specification string and uses
    ``CALENDAR.slice`` to restrict dates to the given range.
    Pass ``None`` for no filtering.
    """
    def __init__(self , value : str | None):
        if value is None:
            self.filter = self.none
        else:
            Logger.alert1(f'filtering date for ModuleData: {value}')

            value = value.strip().replace('-', '~').replace(' ', '~')
            dates = value.split('~')
            assert len(dates) == 2 , f'input.filter.date {value} is not valid , should be yyyyMMdd~yyyyMMdd'
            self.filter = partial(self.slice , start = int(dates[0]) if dates[0] else None , end = int(dates[1]) if dates[1] else None)

        
    def __call__(self , date : np.ndarray) -> np.ndarray:
        """Apply the date filter to ``date`` and return the filtered array."""
        return self.filter(date)

    def filter_blocks(self , blocks : dict[str,DataBlock]) -> dict[str,DataBlock]:
        """Apply the date filter to all blocks' date axis."""
        if not blocks:
            return blocks
        date = self.filter(blocks['y'].date)
        for key,block in blocks.items():
            blocks[key] = block.align_date(date , inplace = True)
        return blocks

    def filter_start_end(self , start : int , end : int) -> tuple[int , int]:
        """
        Return the filtered (start, end) date endpoints.

        Applies ``self.filter`` to the full date range ``[start, end]`` and
        returns ``(first_date, last_date)`` after filtering.  Returns
        ``(99991231, 20070101)`` if no dates remain (empty range).
        """
        dates = self(CALENDAR.range(start , end))
        if len(dates) == 0:
            return 99991231 , 20070101
        return dates[0] , dates[-1]

    @staticmethod
    def none(date : np.ndarray) -> np.ndarray:
        """Pass-through filter; returns ``date`` unchanged."""
        return date

    @staticmethod
    def slice(date : np.ndarray , start : int | None = None , end : int | None = None) -> np.ndarray:
        """Return ``date`` restricted to ``[start, end]`` via ``CALENDAR.slice``."""
        return CALENDAR.slice(date , start , end)