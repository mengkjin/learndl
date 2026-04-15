"""
Loaders for non-standard datasets that do not fit the generic DB → DataBlock pipeline.

Currently the only supported dataset is ``'dfl2'`` (Dongfang L2 characteristics),
which lives in the ``sellside`` database under the ``dongfang.l2_chars`` key.
New special datasets should be added here as additional classmethods and registered
in ``SpecialDataSet.candidates``.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Any

from .data_block import DataBlock , CALENDAR

__all__ = ['SpecialDataSet']

class SpecialDataSet:
    """
    Dispatcher for non-standard DataBlock data sources.

    Each entry in ``candidates`` maps to a dedicated ``load_*`` classmethod.
    Call ``SpecialDataSet.load(key, ...)`` to obtain a ``DataBlock`` for the
    given dataset key.
    """
    candidates : tuple[str,] = ('dfl2' ,)

    @classmethod
    def load(cls , key : str , * , dates : np.ndarray | list[int] | None = None , secid : np.ndarray | None = None , start : int | None = None , end : int | None = None , dtype : str | Any = torch.float , vb_level : Any = 2) -> DataBlock:
        """
        Load a special dataset as a DataBlock.

        Parameters
        ----------
        key : str
            Dataset identifier.  Must be in ``candidates``.
        dates : array-like, optional
            Explicit list of trading dates (yyyyMMdd ints) to load.
            Mutually exclusive with ``start``/``end``.
        secid : np.ndarray, optional
            If provided, align the result to this security universe.
        start, end : int, optional
            Date range (yyyyMMdd) used when ``dates`` is not provided.
        dtype : torch dtype
            Cast the result to this dtype (default: ``torch.float``).
        vb_level : Any
            Verbosity level forwarded to the underlying loader.
        """
        if key == 'dfl2':
            return cls.load_dfl2(dates = dates , secid = secid, start = start , end = end , dtype = dtype , vb_level = vb_level)
        else:
            raise ValueError(f'SpecialModelDataSet {key} is not supported')

    @classmethod
    def load_dfl2(cls , dates : np.ndarray | list[int] | None = None , secid : np.ndarray | None = None , start : int | None = None , end : int | None = None , dtype : str | Any = torch.float , vb_level : Any = 2) -> DataBlock:
        """
        Load Dongfang L2 characteristics (``sellside/dongfang.l2_chars``) as a DataBlock.

        ``dates`` or at least one of ``start``/``end`` must be provided.
        The block is cast to ``dtype`` (default ``torch.float``) and optionally
        re-indexed to ``secid`` via :meth:`DataBlock.align_secid`.
        """
        if dates is None:
            assert start is not None or end is not None , 'dates or start or end must be provided'
            dates = CALENDAR.range(start , end , 'td')
        block = DataBlock.load_raw('sellside', 'dongfang.l2_chars', dates = dates).to(dtype)
        if secid is not None:
            block = block.align_secid(secid , inplace = True)
        return block

    