"""
Basic types alias for the project for type hints
"""
from __future__ import annotations
import numpy as np
from typing import Any , Iterable , Union , Mapping , TypeAlias , Sequence
from pathlib import Path

__all__ = ['strPath' , 'strPaths' , 'intNums']

strPath : TypeAlias = Union[Path , str]
strPaths : TypeAlias = Union[Mapping[int | Any, strPath] , Iterable[strPath]]
intNums : TypeAlias = int | list[int] | np.ndarray[Any, np.dtype[np.int_]] | Sequence[int] | range