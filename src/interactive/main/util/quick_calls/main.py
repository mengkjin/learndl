"""Home page: tutorial, system info, and pending-features banner."""
from __future__ import annotations
import streamlit as st
from src.interactive.frontend import subheader_expander
from .basic import QuickCallButton

PAGE_NAME = 'home'
   
def show_quick_calls(ncol : int = 5) -> None:
    """Show the quick call buttons."""
    with subheader_expander('Quick Call Buttons' , ':material/widgets:' , True , key = 'home-quick-call-buttons'):
        with st.container(key = 'home-quick-call-buttons-container'):
            for i , button in enumerate(QuickCallButton.get_buttons()):
                if i % ncol == 0:
                    cols = st.container().columns(ncol , gap = 'small' , vertical_alignment = 'top')
                with cols[i % ncol]:
                    button.show()