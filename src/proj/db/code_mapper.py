"""
Code mapper tools, used to transform between code and secid.
"""

import pandas as pd
import numpy as np
from typing import Any , Union , TypeVar , Iterable

from src.proj.env import MACHINE

__all__ = ['code2secid' , 'code2code' , 'secid2secid' , 'secid2code']

CodeType = Union[pd.Series , np.ndarray , list[str | int] , str , int]

T = TypeVar("T")

CODE_MAPPER : dict[str , str] = MACHINE.configs('util' , 'transform' , 'mapper' , 'code')
SECID_MAPPER : dict[str , int] = MACHINE.configs('util' , 'transform' , 'mapper' , 'secid')

def _to_series(input : Any) -> pd.Series:
    """Convert input to a pandas series"""
    if isinstance(input , Iterable):
        output = pd.Series([i for i in input])
    else:
        output = pd.Series([input])
    return output

def _from_series(output : pd.Series , input : T) -> T:
    """Convert pandas series to the original input type"""
    if isinstance(input , pd.Series) and type(output) is type(input):
        output.index = input.index
        return output
    elif isinstance(input , Iterable):
        assert isinstance(input , (np.ndarray , list , tuple))
        return input.__class__([o for o in output])
    elif isinstance(input , (int , str)):
        return output.to_numpy().item()
    else:
        raise TypeError(f'Unsupported type of input ({type(input)}) : {input}')

def code2secid(code : T , decode_first = False) -> T:
    """Map tickers to integer security IDs using config mappers and exchange rules.

    Args:
        code: Scalar or vector of symbols in supported input types.
        decode_first: If True, decode bytes-like values before mapping.

    Returns:
        Same container shape/type as ``code`` where possible.
    """
    code_s = _to_series(code)
    code_s = code2code(code_s , decode_first)
    secid = code_s.str.replace('[-.@a-zA-Z]','',regex=True)
    secid = secid.where(secid.str.isdigit() , '-1').astype(int)
    secid = secid2secid(secid)
    assert secid.dtype == int , f'{code_s[secid.isna()].tolist()} cannot be converted to secid'
    secid = _from_series(secid , code)
    return secid

def code2code(code : T , decode_first = False) -> T:
    """Normalize symbols via ``CODE_MAPPER`` from machine config.

    Args:
        code: Scalar or vector of symbols.
        decode_first: If True, decode bytes-like values before mapping.

    Returns:
        Same container shape/type as ``code`` where possible.
    """
    new_code = _to_series(code)
    if decode_first: 
        new_code = pd.Series([(id.decode('utf-8') if isinstance(id , bytes) else str(id)) for id in new_code])
    new_code = new_code.astype(str)
    new_code = new_code.map(CODE_MAPPER).fillna(new_code).astype(str)
    new_code = _from_series(new_code , code)
    return new_code

def secid2secid(secid : T) -> T:
    """Apply ``SECID_MAPPER`` rewrites to integer IDs.

    Returns:
        Same container shape/type as ``secid`` where possible.
    """
    new_secid = _to_series(secid)
    new_secid = new_secid.astype(str).map(SECID_MAPPER).fillna(new_secid).astype(int)
    new_secid = _from_series(new_secid , secid)
    return new_secid

def secid2code(secid : T) -> T:
    """Format secids as zero-padded tickers with ``.SH`` / ``.SZ`` / ``.BJ`` suffixes.

    Returns:
        Same container shape/type as ``secid`` where possible.

    Raises:
        AssertionError: If any secid falls outside supported exchange ranges.
    """
    new_secid = _to_series(secid).astype(int)
    suffix = pd.Series([''] * len(new_secid))
    suffix[(new_secid >= 920000) & (new_secid <= 999999)] = '.BJ'
    suffix[(new_secid >= 830000) & (new_secid <= 899999)] = '.BJ' # some old codes in BJ
    suffix[(new_secid >= 430000) & (new_secid <= 439999)] = '.BJ' # some old codes in BJ
    suffix[(new_secid >=      0) & (new_secid <=  99999)] = '.SZ' # A share in SZSE
    suffix[(new_secid >= 300000) & (new_secid <= 399999)] = '.SZ' # Chuang Ye Ban
    suffix[(new_secid >= 200000) & (new_secid <= 299999)] = '.SZ' # B share in SZSE
    suffix[(new_secid >= 600000) & (new_secid <= 699999)] = '.SH' # A share in SSE
    suffix[(new_secid >= 900000) & (new_secid <= 919999)] = '.SH' # B share in SSE
    assert not suffix.eq('').any(), f'{new_secid[suffix.eq('')].tolist()} is not in the supported range'
    code = new_secid.astype(str).str.zfill(6) + suffix
    code = _from_series(code , secid)
    return code
