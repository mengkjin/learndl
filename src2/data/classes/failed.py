from dataclasses import dataclass
from typing import Optional

@dataclass
class FailedData:
    type: str
    date: Optional[int] = None
    def add_attr(self , key , value): self.__dict__[key] = value