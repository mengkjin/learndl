"""Common type conversion functions for the project"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Any , TypeVar , Mapping , Iterable

from src.proj.core import (
    as_int_array , as_str_array , as_float_array , intDates , intNums , PathsType

)
from .alias import SecidType , DateType , NamesType , npValueType 

__all__ = [
    'ensure_secid' , 'ensure_date' , 'ensure_feature' , 'ensure_name_list' , 
    'ensure_path_list' , 'ensure_int_list' , 'ensure_npvalue'
]

T = TypeVar('T')

def ensure_secid(secid : SecidType , none_as : T = None) -> np.ndarray[Any, np.dtype[np.int_]] | T:
    if secid is None:
        return none_as
    return as_int_array(secid)

def ensure_date(date : DateType | intDates , none_as : T = None) -> np.ndarray[Any, np.dtype[np.int_]] | T:
    if date is None:
        return none_as
    return as_int_array(date)

def ensure_feature(feature : NamesType , none_as : T = None) -> np.ndarray[Any, np.dtype[np.str_]] | T:
    if feature is None:
        return none_as
    return as_str_array(feature)

def ensure_name_list(feature : NamesType , none_as : T = None) -> list[str] | T:
    if feature is None:
        return none_as
    return as_str_array(feature).tolist()

def ensure_path_list(feature : PathsType , none_as : T = None) -> list[Path] | T:
    if feature is None:
        return none_as
    if isinstance(feature, str | Path):
        return [Path(feature)]
    elif isinstance(feature, Mapping):
        return [Path(path) for path in feature.values()]
    elif isinstance(feature, Iterable):
        return [Path(path) for path in feature]
    else:
        raise ValueError(f'Invalid path type: {type(feature)}')

def ensure_int_list(int_list : intNums | None , none_as : T = None) -> list[int] | T:
    if int_list is None:
        return none_as
    return as_int_array(int_list).tolist()

def ensure_npvalue(array : npValueType , none_as : T = None) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | T:
    if array is None:
        return none_as
    return as_float_array(array)