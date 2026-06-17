"""
Streamlit frontend utilities for the interactive application.

action_confirmation
    ``st.dialog``-based confirmation dialogue for destructive actions.

colored_container
    Render a ``st.container`` whose keyed block gets a tinted background via CSS.

expander_subheader
    Render a custom expandable section header with optional help tooltip.
"""
from __future__ import annotations
import html
import streamlit as st
from typing import Any  , Literal , TypeAlias
from collections.abc import Callable

from .basic import unique_st_key

__all__ = ['action_confirmation']

ContainerColor : TypeAlias = Literal['blue' , 'green' , 'red' , 'orange' , 'purple' , 'gray']
ContainerHeight : TypeAlias = Literal['content' , 'stretch'] | int
      
@st.dialog("Please Confirm Your Action")
def action_confirmation(
    on_confirm : Callable[[], None] , on_abort : Callable[[], None] | None = None ,
    title : str = "Are You Sure about This?"
) -> None:
    """``st.dialog``-based confirmation dialogue with Confirm / Abort buttons.

    Parameters
    ----------
    on_confirm:
        Zero-argument callable executed when the user clicks Confirm.
    on_abort:
        Optional zero-argument callable executed when the user clicks Abort.
    title:
        Warning message shown inside the dialogue.
    """
    st.error(f":material/warning:**{title}**")
    col1 , col2 = st.columns(2 , gap = 'small')
    with col1:
        if st.button("**Confirm**" , icon = ":material/check_circle:" , type = "primary" , on_click = on_confirm):
            st.rerun()
    with col2:
        if st.button("**Abort**" , icon = ":material/cancel:" , type = "secondary" , on_click = on_abort):
            st.rerun()


def colored_container(
    color : ContainerColor = 'gray' , key : str | None = None , **kwargs
) -> Any:
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

def subheader_expander(
    label : str , icon : str | None = None , expanded : bool = False ,
    height : ContainerHeight = 'content' , help : str | None = None , status : bool = False , 
    color : str = 'blue' , key : str | None = None
) -> Any:
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
