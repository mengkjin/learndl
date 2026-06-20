"""base types alias for the project for type hints"""
from __future__ import annotations

from typing import Any , Literal , TypeAlias 
from collections.abc import Iterable
from src.proj.core.types import ArrayLike , intDates , PathsType

SingleBenchmark : TypeAlias = str | Any | None
MultipleBenchmark : TypeAlias = Iterable[SingleBenchmark] | SingleBenchmark | Literal['defaults'] | None

SecidType : TypeAlias = ArrayLike[int] | int | None
DateType : TypeAlias = ArrayLike[int] | intDates | None
NamesType : TypeAlias = ArrayLike[str] | tuple[str, ...] | list[str] | str | None
PathsType = PathsType
npValueType : TypeAlias = ArrayLike[float] | float | None
