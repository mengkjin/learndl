from __future__ import annotations
import torch
import numpy as np
from typing import Any

from .data_block import DataBlock , CALENDAR

__all__ = ['SpecialDataSet']

class SpecialDataSet:
    candidates : tuple[str,] = ('dfl2' ,)
    @classmethod
    def load(cls , key : str , * , dates : np.ndarray | list[int] | None = None , secid : np.ndarray | None = None , start : int | None = None , end : int | None = None , dtype : str | Any = torch.float , vb_level : Any = 2) -> DataBlock:
        if key == 'dfl2':
            return cls.load_dfl2(dates = dates , secid = secid, start = start , end = end , dtype = dtype , vb_level = vb_level)
        else:
            raise ValueError(f'SpecialModelDataSet {key} is not supported')

    @classmethod
    def load_dfl2(cls , dates : np.ndarray | list[int] | None = None , secid : np.ndarray | None = None , start : int | None = None , end : int | None = None , dtype : str | Any = torch.float , vb_level : Any = 2) -> DataBlock:
        if dates is None:
            assert start is not None or end is not None , 'dates or start or end must be provided'
            dates = CALENDAR.range(start , end , 'td')
        block = DataBlock.load_raw('sellside', 'dongfang.l2_chars', dates = dates).to(dtype)
        if secid is not None:
            block = block.align_secid(secid , inplace = True)
        return block

    