import itertools , time

import pandas as pd
import numpy as np

from dataclasses import dataclass , field
from typing import Any , Literal

from ..basic import DATAVENDOR , AlphaModel , RISK_MODEL , Portfolio , BENCHMARKS , Benchmark , Port
from ..basic.var import ROUNDING_RETURN , ROUNDING_TURNOVER
from ..optimizer.api import PortfolioOptimizer , PortOptimResult

@dataclass
class PortOptimTuple:
    name : str
    alpha : AlphaModel
    portfolio : Portfolio
    benchmark : Portfolio
    optimizer : PortfolioOptimizer
    lag       : int = 0
    optimrslt : list[PortOptimResult] = field(default_factory=list)
    account   : pd.DataFrame | Any = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha=\'{self.alpha.name}\',benchmark=\'{self.benchmark.name}\',lag={self.lag},'+\
            f'{len(self.portfolio)} fmp\'s,'+'not '* (self.account is None) + 'accounted)'