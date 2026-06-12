"""basic types for the project's data purposes"""
from __future__ import annotations
from typing import Any , Iterable

from src.proj.core.types import StrEnum
from src.proj.cal import Dates

__all__ = ['UpdateType' , 'UpdateFlag' , 'UpdateFlagList']

class UpdateType(StrEnum):
    RECALC = 'recalc'
    UPDATE = 'update'
    ROLLBACK = 'rollback'

class UpdateFlag(StrEnum):
    SUCCESS = 'success'
    FAILED  = 'failed'
    SKIPPED = 'skipped'

    def __bool__(self) -> bool:
        raise NotImplementedError(f'__bool__ is not implemented for {self.__class__.__name__}')

    @property
    def ref_date(self) -> Dates | None:
        if hasattr(self , '_ref_date'):
            return self._ref_date
        return None

    def with_ref_date(self , *args : Any) -> UpdateFlag:
        if args:
            self._ref_date = Dates(args)
        return self

    @property
    def ref_date_str(self) -> str:
        return f'{Dates(self.ref_date)}' if self.ref_date is not None else ''

    def to_list(self) -> UpdateFlagList:
        return UpdateFlagList([self])

    def __add__(self , other : UpdateFlag | Iterable[UpdateFlag]) -> UpdateFlagList:
        return self.to_list() + other

class UpdateFlagList:
    def __init__(self , flags : list[UpdateFlag] | None = None) -> None:
        self.flags : list[UpdateFlag] = flags or []

    def __repr__(self):
        return f'UpdateFlagList({self.flags})'

    def __len__(self):
        return len(self.flags)

    def __bool__(self):
        return len(self) > 0

    def __iter__(self):
        return iter(self.flags)

    def __add__(self , other : UpdateFlag | Iterable[UpdateFlag]):
        self.add(other)
        return self

    def to_list(self):
        return self

    def add(self , flag : UpdateFlag | Iterable[UpdateFlag]) -> None:
        if isinstance(flag , UpdateFlag):
            self.flags.append(flag)
        else:
            self.flags.extend([*flag])

    def summarize(self) -> UpdateFlag:
        if not self.flags or all(flag == UpdateFlag.SKIPPED for flag in self.flags):
            return UpdateFlag.SKIPPED
        if any(flag == UpdateFlag.FAILED for flag in self.flags):
            return UpdateFlag.FAILED
        return UpdateFlag.SUCCESS