"""Home page: tutorial, system info, and pending-features banner."""
from __future__ import annotations
import streamlit as st
from src.api.util.st_frontend import subheader_expander
from .basic import QuickCallButton

__all__ = ['show_quick_calls']

PAGE_NAME = 'home'
   
def show_quick_calls(ncol : int = 5) -> None:
    """Show the quick call buttons."""
    with subheader_expander('Quick Call Buttons' , ':material/widgets:' , True , key = 'home-quick-call-buttons'):
        buttons = QuickCallButton.get_buttons()
        non_research_buttons = [button for button in buttons if not button.research]
        research_buttons = [button for button in buttons if button.research]
        with st.container(key = 'home-quick-call-buttons-container-nonresearch'):
            st.success('Non-research Actions')
            for i , button in enumerate(non_research_buttons):
                if i % ncol == 0:
                    cols = st.container().columns(ncol , gap = 'small' , vertical_alignment = 'top')
                with cols[i % ncol]:
                    button.show()
        with st.container(key = 'home-quick-call-buttons-container-research'):
            st.success('Research Actions')
            for i , button in enumerate(research_buttons):
                if i % ncol == 0:
                    cols = st.container().columns(ncol , gap = 'small' , vertical_alignment = 'top')
                with cols[i % ncol]:
                    button.show()