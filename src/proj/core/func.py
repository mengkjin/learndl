"""
Most fundamental functions for the project.
"""

from __future__ import annotations
import numpy as np
from typing import Any , Iterable

__all__ = ['as_int_array']

def as_int_array(ints: Any) -> np.ndarray[Any, np.dtype[np.int_]]:
    """
    Convert the 'date' parameter of 'td_array' / '*_diff_array' to an integer array index key.
    Single element of this package 'TradeDate' will be expanded to its 'td'; 'pd.Series' will be converted to 'ndarray'.
    """
    if not isinstance(ints, Iterable) or isinstance(ints, str):
        arr = np.atleast_1d(int(ints))
    elif isinstance(ints, np.ndarray):
        arr = ints
    elif ints.__class__.__name__ == 'Series' and hasattr(ints, 'to_numpy'):
        arr = ints.to_numpy() # type: ignore[attr-defined]
    elif ints.__class__.__name__ == 'Tensor':
        arr = ints.cpu().numpy() # type: ignore[attr-defined]
    elif ints.__class__.__name__ == 'Dates' and hasattr(ints, 'dates'):
        arr = ints.dates # type: ignore[attr-defined]
    elif isinstance(ints, Iterable):
        arr = np.atleast_1d(np.asarray(ints))
    else:
        raise ValueError(f'Invalid dates type: {type(ints)} {ints}')
    assert isinstance(arr, np.ndarray), f'dates is not np.ndarray: {type(arr)}'
    return arr.astype(int)