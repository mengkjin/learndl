import numpy as np
import pandas as pd
import xarray as xr  

from dataclasses import dataclass
from torch import Tensor
from typing import Optional

@dataclass
class FailedData:
    type: str
    date: Optional[int] = None
    def add_attr(self , key , value): self.__dict__[key] = value

@dataclass(slots=True)
class NdData:
    values : np.ndarray | Tensor 
    index  : list | tuple
    def __post_init__(self):
        assert self.values.ndim == len(self.index) , (self.values.ndim , len(self.index))

    def __repr__(self):
        return '\n'.join([str(self.__class__) , f'values shape {self.shape}'])

    @property
    def shape(self): return self.values.shape
    @property
    def ndim(self): return self.values.ndim

    @classmethod
    def from_xarray(cls , xarr : xr.Dataset):
        values = np.stack([arr.to_numpy() for arr in xarr.data_vars.values()] , -1)
        index = [arr.values for arr in xarr.indexes.values()] + [list(xarr.data_vars)]
        return cls(values , index)

    @classmethod
    def from_dataframe(cls , df : pd.DataFrame):
        index = [l.values for l in df.index.levels] + [df.columns.values] #type:ignore
        if len(df) != len(index[0]) * len(index[1]): 
            return cls.from_xarray(xr.Dataset.from_dataframe(df))
        else:
            values = df.values.reshape(len(index[0]) , len(index[1]) , -1)
            return cls(values , index)