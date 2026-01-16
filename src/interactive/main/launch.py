

import pathlib , os
import streamlit as st
import streamlit_autorefresh as st_autorefresh

from src.proj import Proj
from util.navigation import page_setup

file_path = str(pathlib.Path(__file__).absolute())
path = file_path.removesuffix(file_path.split('learndl' , 1)[-1]).lower()

assert os.getcwd().lower() == path , \
    f'current working directory {os.getcwd()} is not {path} , do not know where to find src file'

st.set_option('client.showSidebarNavigation', False)

if Proj.Conf.Interactive.auto_refresh_interval > 0:
    st_autorefresh.st_autorefresh(interval = Proj.Conf.Interactive.auto_refresh_interval * 1000)
  
def main():
    Proj.print_info(script_level = False)
    pg = page_setup()
    pg.run()
    
if __name__ == '__main__':
    main() 