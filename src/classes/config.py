from dataclasses import dataclass
from typing import Optional

@dataclass(slots=True)
class DataProcessCfg:
    db_src  : str
    db_key  : str | list
    feature : Optional[list] = None

    def __post_init__(self):
        if isinstance(self.db_key , str): self.db_key = [self.db_key]