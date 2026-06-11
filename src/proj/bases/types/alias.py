"""base types alias for the project for type hints"""
from __future__ import annotations

from typing import Any , Literal , TypeAlias , Iterable

from src.proj.cal.cal import TradeDate , intDate , intDateNone , intDates
from src.proj.core.types.alias import strPath , strPaths

__all__ = [
    'strPath' , 'strPaths' , 'TradeDate' , 'intDate' , 'intDateNone' , 'intDates' , 
    'MultipleBenchmark' , 'SingleBenchmark']

SingleBenchmark : TypeAlias = str | Any | None
MultipleBenchmark : TypeAlias = Iterable[str | Any | None] | str | Literal['defaults'] | None