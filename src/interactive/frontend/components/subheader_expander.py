"""
Streamlit frontend utilities for the interactive application.

expander_subheader
    Render a custom expandable section header with optional help tooltip.
"""
from __future__ import annotations
import html
from typing import Literal , Any
import streamlit as st

from .basic import unique_st_key


def subheader_expander(label : str , icon : str | None = None , expanded : bool = False ,
                       height : int | Literal['content' , 'stretch'] = 'content' , help : str | None = None , status : bool = False , 
                       color : str = 'blue' , key : str | None = None) -> Any:
    """Render a custom collapsible section header with an optional help tooltip.

    When *help* is set and ``status=False``, the help HTML is rendered **inside**
    the same keyed ``st.container`` as the expander so global CSS in
    ``templates/css/interactive/subheader_expander.template`` can show
    ``.help-tooltip`` while ``summary`` is hovered (``:has(...)``).

    Parameters
    ----------
    
    label:
        Header text displayed on the expander button.
    icon:
        Optional Material icon string prepended to *label*.
    expanded:
        Whether the section starts in the open state.
    height:
        Inner container height (px, ``'content'``, or ``'stretch'``).
    help:
        Help body (HTML-escaped). For ``st.status``, uses ``st.markdown(..., help=...)``.
    status:
        If True, use ``st.status`` instead of ``st.expander``.
    color:
        Streamlit badge colour for the label.
    key:
        Unique key used to build the Streamlit container key. If not provided, a unique key will be generated.

    Returns
    -------
    Streamlit container
        The inner container that callers should use as a context manager.
    """
    unique_key = unique_st_key(key, 'subheader-expander')
    container = st.container(key = unique_key)
    with container:
        if help is not None:
            st.markdown(
                f'<div class="expander-help-container"><div class="help-tooltip">{html.escape(help)}</div></div>' ,
                unsafe_allow_html = True ,
            )
        full_label = label if icon is None else f'{icon} {label}'
        if status:
            exp_container = container.status(f" :{color}[{full_label}]" , expanded = expanded)
        else:
            exp_container = container.expander(f":{color}[{full_label}]" , expanded = expanded).container(height = height)

    return exp_container
