"""
N-dimensional array wrapper pairing a tensor/ndarray with per-axis coordinate arrays.

Used internally by DataBlock.from_pandas and DataBlock.from_polars as an intermediate
representation when converting a long-format DataFrame into a structured multi-dimensional
array before constructing the final 4-D tensor.
"""

import torch
import numpy as np
import pandas as pd
import xarray as xr

from dataclasses import dataclass
@dataclass(slots=True)
class NdData:
    """
    Generic n-dimensional data container: a values array plus one coordinate array per axis.

    Attributes
    ----------
    values : np.ndarray | torch.Tensor
        The underlying data array of shape ``(d0, d1, ..., dn)``.
    index : list[np.ndarray]
        One coordinate array per axis, so ``len(index) == values.ndim``.
    """
    values : np.ndarray | torch.Tensor 
    index  : list[np.ndarray]
    def __post_init__(self):
        """Validate shape consistency and normalise index elements to np.ndarray."""
        assert self.values.ndim == len(self.index) , (self.values.ndim , len(self.index))
        self.index = [(ii if isinstance(ii , np.ndarray) else np.array(ii)) for ii in self.index]

    def __repr__(self):
        """Return a human-readable multi-line summary of shape, finite ratio, and index."""
        return '\n'.join([str(self.__class__) ,
                          f'values shape : {self.shape}' ,
                          f'finite ratio : {self.finite_ratio():.4f}' ,
                          f'index : {str(self.index)}'])

    def __len__(self):
        """Return the length of the first axis."""
        return self.shape[0]

    @property
    def shape(self):
        """Shape of the underlying values array."""
        return self.values.shape

    @property
    def ndim(self):
        """Number of dimensions of the underlying values array."""
        return self.values.ndim

    def finite_ratio(self) -> float:
        """Return the fraction of finite (non-NaN, non-Inf) elements in values."""
        if isinstance(self.values , np.ndarray):
            n_finite = np.isfinite(self.values).sum()
            n_total = self.values.size
        else:
            n_finite = self.values.isfinite().sum().item()
            n_total = self.values.numel()
        return n_finite / n_total

    @classmethod
    def from_xarray(cls , xarr : xr.Dataset):
        """
        Construct NdData from an xarray Dataset.

        Data variables become the last axis; Dataset dimension coordinates become
        the preceding axes in order.
        """
        values = np.stack([arr.to_numpy() for arr in xarr.data_vars.values()] , -1)
        index = [arr.values for arr in xarr.indexes.values()] + [list(xarr.data_vars)]
        return cls(values , index)

    @classmethod
    def from_dataframe(cls , df : pd.DataFrame):
        """
        Construct NdData from a pandas DataFrame with a MultiIndex.

        If the MultiIndex forms a complete Cartesian product the values are
        reshaped directly (fast path). Otherwise the data is routed through
        ``xr.Dataset.from_dataframe`` to handle the sparse/ragged case.
        """
        index = [lvl.values for lvl in df.index.levels] + [df.columns.values] #type:ignore
        if len(df) != len(index[0]) * len(index[1]):
            return cls.from_xarray(xr.Dataset.from_dataframe(df))
        else:
            values = df.values.reshape(len(index[0]) , len(index[1]) , -1)
            return cls(values , index)