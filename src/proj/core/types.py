from __future__ import annotations
from typing import Any , Iterable , Union , Mapping , TypeAlias
from pathlib import Path

__all__ = ['strPath' , 'strPaths']

strPath : TypeAlias = Union[Path , str]
strPaths : TypeAlias = Union[Mapping[int | Any, strPath] , Iterable[strPath]]