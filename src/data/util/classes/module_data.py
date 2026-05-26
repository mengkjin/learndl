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
from __future__ import annotations
import torch
import numpy as np

from copy import deepcopy
from dataclasses import dataclass
from functools import partial, cached_property
from typing import Any, Literal

from src.proj import Logger, CALENDAR
from src.proj.util import properties, BaseModule

from .data_block import DataBlock, DataBlockNorm, data_type_abbr
from .special_dataset import SpecialDataSet

__all__ = ["ModuleData"]
@dataclass
class ModuleDataConfig:
    data_type_list: tuple[str, ...]
    y_labels: list[str] | None
    use_data: Literal["fit", "predict", "both"]
    factor_names: list[str] | None
    factor_start_dt: int | None
    factor_end_dt: int | None
    filter_secid: str | None
    filter_date: str | None
    dtype: torch.dtype

    @property
    def fit_for_cache(self):
        return (bool(self.data_type_list) and
            self.filter_secid is None and
            self.filter_date is None and
            self.use_data in ['fit' , 'both']
        )

    def covered_by(self, other: ModuleDataConfig) -> bool:
        return (
            self.data_type_list == other.data_type_list and
            self.y_labels == other.y_labels and
            (self.use_data == other.use_data or other.use_data == 'both') and
            self.factor_names == other.factor_names and
            self.factor_start_dt == other.factor_start_dt and
            self.factor_end_dt == other.factor_end_dt and
            self.filter_secid == other.filter_secid and
            self.filter_date == other.filter_date and
            self.dtype == other.dtype
        )

    @staticmethod
    def get_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
        if dtype is None:
            return torch.float
        if isinstance(dtype, str):
            return getattr(torch, dtype)
        return dtype
class ModuleData(BaseModule):
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
        Verbosity / indentation for the module messages.
    dtype : torch dtype
        Dtype to cast loaded blocks to (default: ``torch.float``).
    """
    def __init__(
        self,
        config: ModuleDataConfig,
        indent: int = 1,
        vb_level: Any = 2
    ):
        self.config = config
        self.set_vb(vb_level , indent)

    @classmethod
    def initialize(cls, data_type_list: list[str],
        y_labels: list[str] | None = None,
        use_data: Literal["fit", "predict", "both"] = "fit",
        *,
        factor_names: list[str] | None = None,
        factor_start_dt: int | None = None,
        factor_end_dt: int | None = None,
        filter_secid: str | None = None,
        filter_date: str | None = None,
        indent: int = 1,
        vb_level: Any = 2,
        dtype=torch.float) -> ModuleData:
        config = ModuleDataConfig(
            data_type_list=tuple(sorted([cls.abbr(data_type) for data_type in data_type_list])),
            y_labels=y_labels,
            use_data=use_data,
            factor_names=factor_names,
            factor_start_dt=factor_start_dt,
            factor_end_dt=factor_end_dt,
            filter_secid=filter_secid,
            filter_date=filter_date,
            dtype=ModuleDataConfig.get_torch_dtype(dtype)
        )
        return cls(config = config, indent = indent, vb_level = vb_level)

    @property
    def data_type_list(self):
        return list(self.config.data_type_list)

    @property
    def y_labels(self):
        return self.config.y_labels

    @property
    def use_data(self):
        return self.config.use_data

    @property
    def factor_names(self):
        return self.config.factor_names

    @property
    def load_keys(self):
        """All block keys to be loaded: ``['y', *data_type_list]``."""
        return ["y", *self.data_type_list]

    @cached_property
    def blocks(self) -> dict[str, DataBlock]:
        return {}

    @cached_property
    def norms(self) -> dict[str, DataBlockNorm]:
        return {}

    @cached_property
    def secid_filter(self):
        return SecidFilter(self.config.filter_secid if self.use_data == "fit" else None)
    
    @cached_property
    def date_filter(self):
        return DateFilter(self.config.filter_date if self.use_data == "fit" else None)

    @cached_property
    def x(self) -> dict[str, DataBlock]:
        """All blocks except ``'y'`` (the feature/input blocks)."""
        return {key: value for key, value in self.blocks.items() if key != "y"}

    @cached_property
    def y(self) -> DataBlock:
        """The label block, feature-aligned to ``y_labels``."""
        return self.blocks["y"].align_feature(self.y_labels)

    @property
    def empty_x(self):
        """True if no feature blocks are loaded or all are empty."""
        return len(self.x) == 0 or all([x.empty for x in self.x.values()])

    @property
    def shape(self):
        """Shape summary across x, y, secid, and date."""
        return properties.shape(self, ["x", "y", "secid", "date"])

    @property
    def secid(self):
        """Security universe from the ``'y'`` block."""
        return self.blocks["y"].secid

    @property
    def date(self):
        """Date range from the ``'y'`` block."""
        return self.blocks["y"].date

    @cached_property
    def loaded(self):
        """True after ``load()`` has been called successfully."""
        return False

    @property
    def block_title(self):
        """Human-readable summary of the block set for log messages."""
        return (
            f"{len(self.load_keys)} DataBlocks"
            if len(self.load_keys) > 4
            else f"DataBlock [{','.join(self.load_keys)}]"
        )

    def __bool__(self):
        """True when at least one non-empty feature block is loaded."""
        return not self.empty_x

    def copy(self):
        """Return a deep copy of this ModuleData instance."""
        return deepcopy(self)

    def date_within(self, start: int, end: int, interval=1) -> np.ndarray:
        """Return the loaded date array sliced to [start, end] with optional stride."""
        return CALENDAR.slice(self.date, start, end)[::interval]

    def target_start_end(self):
        """
        Compute the (start, end) date range to load data for.

        - ``'predict'`` mode: loads the last 366 calendar days up to today.
        - ``'fit'`` mode: loads from 20070101 up to the last available ``'y'`` dump date.
        """
        start = (
            CALENDAR.td(CALENDAR.updated(), -366).td
            if self.use_data == "predict"
            else 20070101
        )
        end = (
            DataBlock.max_data_date("y", "fit")
            if self.use_data == "fit"
            else CALENDAR.updated()
        )
        end = end or CALENDAR.updated()
        return start, end

    def load(self):
        """
        load all relevant data of this module data, should be called before any other operations
        blocks: ['y' , *data_type_list] DataBlocks
        norms: ['y' , *data_type_list] DataBlockNorms (if exists)
        factor: factor_names DataBlock
        """
        if self.loaded:
            return self
        self._init_data()
        self._load_blocks()
        self._align_blocks()
        self._load_norms()
        self._load_factor()
        DataBlock.blocks_ffill(self.blocks, exclude=["y"])
        self.loaded = True
        return self

    def _init_data(self):
        self.__dict__.pop("x", None)
        self.__dict__.pop("y", None)
        self.blocks.clear()
        self.norms.clear()
        return self

    def _load_blocks(self):
        """
        load all blocks including y and data_type_list blocks.

        Iterates over ``load_keys`` (``'y'`` first, then X blocks).
        The ``'y'`` block drives the secid and date grids that all X blocks
        are aligned to in the subsequent ``align_blocks()`` call.
        """
        start , end = self.target_start_end()
        date = self.date_filter(CALENDAR.range(start , end))
        for i , key in enumerate(self.load_keys):
            assert i > 0 or key == 'y' , f'{key} must be the first key'
            self.blocks[key] = self.load_one(key, dates = date)
        return self

    def _align_blocks(self):
        """Align all loaded blocks to a common (secid, date) grid via ``DataBlock.blocks_align``."""
        if not self.blocks:
            return self
        date = self.date_filter(CALENDAR.range(*self.target_start_end()))
        secid = self.secid_filter(self.secid)
        for key , block in self.blocks.items():
            self.blocks[key] = block.align_secid_date(secid , date , inplace = True)
        index_lens = [block.shape[:2] for block in self.blocks.values()]
        if index_lens:
            assert all([lens == (len(secid), len(date)) for lens in index_lens]), (
                f"{[(name, block.shape) for name, block in self.blocks.items()]}"
            )
        return self

    def load_one(
        self, key: str, *, dates: np.ndarray, secid: np.ndarray | None = None, **kwargs
    ):
        """
        Load a single block for the given key and date array.

        Dispatches to ``load_preprocess_block`` for registered preprocessors
        or ``load_special_block`` for special dataset candidates.
        Returns an empty ``DataBlock`` when ``dates`` is empty.
        """
        if len(dates) == 0:
            return DataBlock()
        from src.data.preprocess import PrePros
        if key in PrePros.keys():
            return self.load_preprocess_block(
                key, dates=dates, secid=secid, vb_level=self.vb_level + 2, **kwargs
            )
        elif key in SpecialDataSet.candidates:
            return self.load_special_block(
                key, dates=dates, secid=secid, vb_level=self.vb_level + 2, **kwargs
            )
        else:
            raise ValueError(f"key [{key}] is not supported")

    def load_preprocess_block(
        self, key: str, *, dates: np.ndarray, secid: np.ndarray | None = None, **kwargs
    ):
        """Load a block via the registered ``PreProcessor`` for ``key``."""
        from src.data.preprocess import PrePros
        type = "predict" if self.use_data == "predict" else "fit"
        block = PrePros.get_processor(key, type, self.indent + 1, self.vb_level + 2).load(dates, secid=secid)
        return block

    def load_special_block(
        self, key: str, *, dates: np.ndarray, secid: np.ndarray | None = None, **kwargs
    ):
        """Load a block via ``SpecialDataSet`` for non-standard dataset keys."""
        block = SpecialDataSet.load(
            key, dates=dates, secid=secid, dtype=self.config.dtype, vb_level=self.vb_level + 2
        )
        return block

    def _load_norms(self):
        """Load ``DataBlockNorm`` objects for all X block types from disk (no-op if already loaded)."""
        if self.norms:
            return
        self.norms.update(
            DataBlock.load_preprocess_norms(self.data_type_list, dtype=self.config.dtype)
        )

    def _load_factor(self):
        """load factor data"""
        if not self.factor_names:
            return self
        factor_title = (
            f"{len(self.factor_names)} Factors"
            if len(self.factor_names) > 1
            else f"Factor [{self.factor_names[0]}]"
        )
        start = max(self.config.factor_start_dt or self.date[0], self.date[0])
        end = min(self.config.factor_end_dt or self.date[-1], self.date[-1])
        with self.logger.timer(f"Load {factor_title} ({start} - {end})"):
            from src.data.loader import FactorLoader
            self.blocks["factor"] = FactorLoader(self.factor_names , vb_level="never").\
                load(start, end).align_secid_date(self.secid, self.date, inplace=True)
        return self

    @staticmethod
    def abbr(data_type: str):
        """Normalise a data-type key via ``data_type_abbr``."""
        return data_type_abbr(data_type)

    @classmethod
    def min_data_date(
        cls, data_type_list: list[str], factor_names: list[str] | None = None
    ) -> int | None:
        """Return the minimum data date from the loaded blocks and factor names."""
        from src.res.factor.calculator import StockFactorHierarchy

        dates: list[int] = []
        for data_type in ["y", *data_type_list]:
            fit_min_date = DataBlock.min_data_date(data_type, "fit")
            predict_min_date = DataBlock.min_data_date(data_type, "predict")
            if fit_min_date is None and predict_min_date is None:
                return None
            dates.append(min(fit_min_date or 99991231, predict_min_date or 99991231))
        if factor_names:
            for factor_name in factor_names:
                dates.append(StockFactorHierarchy.get_factor(factor_name).min_date)
        if len(dates) == 0:
            return None
        else:
            return CALENDAR.td_array([max(dates)], backward=False)[0]

    @classmethod
    def max_data_date(
        cls, data_type_list: list[str], factor_names: list[str] | None = None
    ) -> int | None:
        """Return the maximum data date from the loaded blocks and factor names."""
        from src.res.factor.calculator import StockFactorHierarchy

        dates: list[int] = []
        for data_type in data_type_list:
            fit_max_date = DataBlock.max_data_date(data_type, "fit")
            predict_max_date = DataBlock.max_data_date(data_type, "predict")
            if fit_max_date is None and predict_max_date is None:
                return None
            dates.append(max(fit_max_date or 19000101, predict_max_date or 19000101))
        if factor_names:
            for factor_name in factor_names:
                dates.append(StockFactorHierarchy.get_factor(factor_name).max_date)
        if len(dates) == 0:
            return None
        else:
            return CALENDAR.td_array([max(dates)], backward=True)[0]


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

    def __init__(self, value: str | None):
        if value is None:
            self.filter = self.none
        else:
            Logger.alert1(f"filtering secid for ModuleData: {value}", vb_level=2)
            if value.startswith("random."):
                self.filter = partial(self.random, num=int(value.split(".")[1]))
            elif value.startswith("first."):
                self.filter = partial(self.first, num=int(value.split(".")[1]))
            elif value in ["csi300", "csi500", "csi1000"]:
                self.filter = partial(self.benchmark, bm=value)
            else:
                raise ValueError(
                    f"input.filter.secid {value} is not valid , should be random.200 , first.200 , csi300 , csi500 , csi1000"
                )

    def __call__(self, secid: np.ndarray) -> np.ndarray:
        """Apply the configured filter to ``secid`` and return the filtered array."""
        return self.filter(secid)

    def filter_block(self, block: DataBlock , inplace: bool = True) -> DataBlock:
        """Apply the filter to a single block."""
        return block.align_secid(self.filter(block.secid), inplace=inplace)

    def filter_blocks(self, blocks: dict[str, DataBlock] , inplace: bool = True) -> dict[str, DataBlock]:
        """Apply the filter to all blocks in the dict (aligns secid axis of each block)."""
        if not blocks:
            return blocks
        secid = self.filter(blocks["y"].secid)
        for key, block in blocks.items():
            blocks[key] = block.align_secid(secid, inplace=inplace)
        return blocks

    @staticmethod
    def none(secid: np.ndarray) -> np.ndarray:
        """Pass-through filter; returns ``secid`` unchanged."""
        return secid

    @staticmethod
    def random(secid: np.ndarray, num: int) -> np.ndarray:
        """Return a random sample of ``num`` secids without replacement."""
        return np.random.choice(secid, num, replace=False)

    @staticmethod
    def first(secid: np.ndarray, num: int) -> np.ndarray:
        """Return the first ``num`` secids in the input array."""
        return secid[:num]

    @classmethod
    def Benchmark(cls):
        """Lazy accessor for the ``Benchmark`` class (avoids circular import)."""
        if not hasattr(cls, "_benchmark"):
            from src.res.factor.util.classes.benchmark import Benchmark

            cls._benchmark = Benchmark
        return cls._benchmark

    @classmethod
    def benchmark(cls, secid: np.ndarray, bm: str, date: int = 20200104) -> np.ndarray:
        """Return the constituent secids of benchmark ``bm`` as of ``date``."""
        return cls.Benchmark()(bm).get(date, True).secid


class DateFilter:
    """
    Callable filter that subsets a date array.

    Parses a ``'yyyyMMdd~yyyyMMdd'`` specification string and uses
    ``CALENDAR.slice`` to restrict dates to the given range.
    Pass ``None`` for no filtering.
    """

    def __init__(self, value: str | None):
        if value is None:
            self.filter = self.none
        else:
            Logger.alert1(f"filtering date for ModuleData: {value}", vb_level=2)

            value = value.strip().replace("-", "~").replace(" ", "~")
            dates = value.split("~")
            assert len(dates) == 2, (
                f"input.filter.date {value} is not valid , should be yyyyMMdd~yyyyMMdd"
            )
            self.filter = partial(
                self.slice,
                start=int(dates[0]) if dates[0] else None,
                end=int(dates[1]) if dates[1] else None,
            )

    def __call__(self, date: np.ndarray) -> np.ndarray:
        """Apply the date filter to ``date`` and return the filtered array."""
        return self.filter(date)

    def filter_block(self, block: DataBlock , inplace: bool = True) -> DataBlock:
        """Apply the date filter to a single block."""
        return block.align_date(self.filter(block.date), inplace=inplace)

    def filter_blocks(self, blocks: dict[str, DataBlock] , inplace: bool = True) -> dict[str, DataBlock]:
        """Apply the date filter to all blocks' date axis."""
        if not blocks:
            return blocks
        date = self.filter(blocks["y"].date)
        for key, block in blocks.items():
            blocks[key] = block.align_date(date, inplace=inplace)
        return blocks

    def filter_start_end(self, start: int, end: int) -> tuple[int, int]:
        """
        Return the filtered (start, end) date endpoints.

        Applies ``self.filter`` to the full date range ``[start, end]`` and
        returns ``(first_date, last_date)`` after filtering.  Returns
        ``(99991231, 20070101)`` if no dates remain (empty range).
        """
        dates = self(CALENDAR.range(start, end))
        if len(dates) == 0:
            return 99991231, 20070101
        return dates[0], dates[-1]

    @staticmethod
    def none(date: np.ndarray) -> np.ndarray:
        """Pass-through filter; returns ``date`` unchanged."""
        return date

    @staticmethod
    def slice(
        date: np.ndarray, start: int | None = None, end: int | None = None
    ) -> np.ndarray:
        """Return ``date`` restricted to ``[start, end]`` via ``CALENDAR.slice``."""
        return CALENDAR.slice(date, start, end)
