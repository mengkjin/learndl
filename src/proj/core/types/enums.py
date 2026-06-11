from __future__ import annotations
import enum
from typing import Any , Iterable , Self

__all__ = ['StrEnum']

class StrEnum(enum.StrEnum):
    """Custom string enum"""

    @classmethod
    def values(cls) -> tuple[str,...]:
        return tuple(cls.value for cls in cls)

    @classmethod
    def ensure_list(cls , x : Any | Iterable) -> list[Self]:
        if x in cls:
            ret = [x]
        elif isinstance(x , list):
            ret = x
        else:
            ret = list(cls)
        ret = [cls(i) for i in ret]
        return ret