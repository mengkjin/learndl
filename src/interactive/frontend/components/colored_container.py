"""
Streamlit frontend utilities for the interactive application.

colored_container
    Render a ``st.container`` whose keyed block gets a tinted background via CSS.
"""
from __future__ import annotations
from typing import Any , Literal
import streamlit as st

from .basic import unique_st_key

def colored_container(color : Literal['blue' , 'green' , 'red' , 'orange' , 'purple' , 'gray'] = 'gray' , key : str | None = None , **kwargs) -> Any:
    """Render a custom container with a colored background.

    Parameters
    ----------
    color:
        background color for the container , Any valid CSS color.
    key:
        Unique key used to build the Streamlit container key. If not provided, a unique key will be generated.

    Returns
    -------
    Streamlit container
        The inner container that callers should use as a context manager.
    """
    unique_key = unique_st_key(key, f'colored-container-{color}')
    container = st.container(key = unique_key , **kwargs)
    return container
