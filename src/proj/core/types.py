from typing import Any , Iterable , Union , Mapping , TypeAlias
from pathlib import Path

strPath : TypeAlias = Union[Path , str]
strPaths : TypeAlias = Union[Mapping[int | Any, strPath] , Iterable[strPath]]