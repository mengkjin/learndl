"""
Streamlit frontend utilities for the interactive application.

basic
    Basic utilities for the interactive application.
"""
from __future__ import annotations
import uuid
import streamlit as st

def unique_st_key(key : str | None = None, custom_type : str | None = None):
    """Generate a unique Streamlit key based on the current session and the provided root key."""
    key = key.replace(" " , "-").lower() if key else 'custom'
    if custom_type:
        key += f'-{custom_type}'
    unique_key = key
    while unique_key in st.session_state:
        unique_key = f'{key}-{uuid.uuid4()}'
    return unique_key