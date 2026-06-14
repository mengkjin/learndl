"""Application entry point for the Learndl interactive pipeline manager.

Validates that the process is launched from the project root, applies optional
auto-refresh, and delegates to :func:`page_setup` which assembles navigation
and renders each page.

Run with::

    uv run streamlit run src/api/interactive/launch.py
"""

from __future__ import annotations
import pathlib
import streamlit as st
import streamlit_autorefresh as st_autorefresh

from src.proj import Proj , Const
from src.api.interactive.util.navigation import page_setup

__all__ = ['main']

file_path = str(pathlib.Path(__file__).absolute())

st.set_option('client.showSidebarNavigation', False)

if (auto_refresh_interval := Const.Pref.interactive.get('auto_refresh_interval' , 0)) > 0:
    st_autorefresh.st_autorefresh(interval = auto_refresh_interval * 1000)
  
def main() -> None:
    """Bootstrap and run the Streamlit app.

    Prints project info, builds the multi-page navigation object, and runs it.
    """
    Proj.print_info(once_type = 'os')
    pg = page_setup()
    pg.run()
    
if __name__ == '__main__':
    main() 