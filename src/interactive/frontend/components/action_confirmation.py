"""
Streamlit frontend utilities for the interactive application.

action_confirmation
    ``st.dialog``-based confirmation dialogue for destructive actions.
"""
from __future__ import annotations
from typing import Callable
import streamlit as st
      
@st.dialog("Please Confirm Your Action")
def action_confirmation(on_confirm : Callable[[], None] , on_abort : Callable[[], None] | None = None ,
                        title : str = "Are You Sure about This?") -> None:
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