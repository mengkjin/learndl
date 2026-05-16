"""Container for trainer intermediate results"""

from __future__ import annotations

import pandas as pd
from collections import defaultdict

class TypedContainer:
    """Container for trainer intermediate results"""
    def __init__(self):
        self.dataframes = defaultdict(pd.DataFrame)

    def __repr__(self):
        return f'TrainerContainer(dataframes={len(self.dataframes)})'