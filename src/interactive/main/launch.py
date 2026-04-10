

import pathlib , os
import streamlit as st
import streamlit_autorefresh as st_autorefresh

from src.proj import Proj , CONST
from src.interactive.main.util.navigation import page_setup

file_path = str(pathlib.Path(__file__).absolute())
path = file_path.removesuffix(file_path.split('learndl' , 1)[-1]).lower()

assert os.getcwd().lower() == path , \
    f'current working directory {os.getcwd()} is not {path} , do not know where to find src file'

st.set_option('client.showSidebarNavigation', False)

if CONST.Pref.get('interactive' , 'auto_refresh_interval' , 0) > 0:
    st_autorefresh.st_autorefresh(interval = CONST.Pref.get('interactive' , 'auto_refresh_interval' , 0) * 1000)
  
def main():
    Proj.print_info(script_level = False)
    pg = page_setup(navigation_position = 'both')
    pg.run()
    
if __name__ == '__main__':
    main() 