"""base types for the project, only used for type hints, do not import anything from this module outside of TYPE_CHECKING"""
from __future__ import annotations

from typing import Union , TypeAlias , Sequence , TYPE_CHECKING

__all__ = ['strPath' , 'strPaths' , 'TradeDate' , 'intDate' , 'intDateNone' , 'intDates']

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from src.proj.cal import TradeDate
    from src.proj.core.types import strPath , strPaths
    intDate : TypeAlias = Union[int , TradeDate]
    intDateNone : TypeAlias = Union[int, TradeDate, None]
    intDates : TypeAlias = Union[intDate , Sequence[intDate], np.ndarray , pd.Series]