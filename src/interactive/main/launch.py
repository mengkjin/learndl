"""Application entry point for the Learndl interactive pipeline manager.

Validates that the process is launched from the project root, applies optional
auto-refresh, and delegates to :func:`page_setup` which assembles navigation
and renders each page.

Run with::

    uv run streamlit run src/interactive/main/launch.py
"""

import pathlib , os
import streamlit as st
import streamlit_autorefresh as st_autorefresh

from src.proj import Proj , Const
from src.interactive.main.util.navigation import page_setup

file_path = str(pathlib.Path(__file__).absolute())
path = file_path.removesuffix(file_path.split('learndl' , 1)[-1]).lower()

assert os.getcwd().lower() == path , \
    f'current working directory {os.getcwd()} is not {path} , do not know where to find src file'

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