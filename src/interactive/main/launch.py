

import pathlib , os
import streamlit as st
import streamlit_autorefresh as st_autorefresh

from src.proj import Proj
from util import style
from util.navigation import page_navigation , sidebar_navigation

file_path = str(pathlib.Path(__file__).absolute())
path = file_path.removesuffix(file_path.split('learndl' , 1)[-1]).lower()

assert os.getcwd().lower() == path , \
    f'current working directory {os.getcwd()} is not {path} , do not know where to find src file'

st.set_option('client.showSidebarNavigation', False)

if Proj.Conf.Interactive.auto_refresh_interval > 0:
    st_autorefresh.st_autorefresh(interval = Proj.Conf.Interactive.auto_refresh_interval * 1000)

def page_config():
    st.set_page_config(
        page_title=Proj.Conf.Interactive.page_title,
        page_icon=":material/rocket_launch:",
        layout= 'wide' , # 'centered',
        initial_sidebar_state="expanded"
    )
    style()
    #st.session_state['box-title'].title(f":rainbow[:material/rocket_launch: {__page_title__} (_v{__version__}_)]")
    #if cols[1].button(':rainbow[:material/home:]' , key = 'go-home-button' , help = 'Go to Home Page'): 
    #    st.switch_page(get_intro_page('home')['page'])
  
def main():
    Proj.print_info(script_level = False)
    page_config()
    pg = page_navigation()
    sidebar_navigation()
    pg.run()
    
if __name__ == '__main__':
    main() 