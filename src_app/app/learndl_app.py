

import sys , pathlib , os
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl' , 1)[-1])
if not path in sys.path: sys.path.append(path)
assert os.getcwd() == path , \
    f'current working directory {os.getcwd()} is not {path} , do not know where to find src file'

import streamlit as st
import streamlit_autorefresh as st_autorefresh

from util import (__page_title__ , __version__ , AUTO_REFRESH_INTERVAL , 
                  style , menu_pages , script_pages)

st.set_option('client.showSidebarNavigation', False)

if AUTO_REFRESH_INTERVAL > 0:
    st_autorefresh.st_autorefresh(interval = AUTO_REFRESH_INTERVAL * 1000)

def page_config():
    st.set_page_config(
        page_title=__page_title__,
        page_icon=":material/rocket_launch:",
        layout='wide',
        initial_sidebar_state="expanded"
    )
    style()
    st.title(f":rainbow[:material/rocket_launch: {__page_title__} (_v{__version__}_)]")
    

def page_navigation():
    pages = {}
    pages['Introduction'] = [page['page'] for page in menu_pages().values()]
    for page in script_pages().values():
        group_name = page['group'].title() + ' Scripts'
        if group_name not in pages: pages[group_name] = []
        pages[group_name].append(page['page'])
    pg = st.navigation(pages = pages , position='top')
    return pg

def sidebar_navigation():
    with st.sidebar:
        #st.session_state['sidebar-runner-header'] = st.empty()
        st.session_state['sidebar-runner-button'] = st.empty()
        st.subheader(":blue[:material/file_present: Script Shortcuts]")
        with st.container(key = "sidebar-script-menu"):
            group = ''
            for page in script_pages().values():
                if page['group'] != group:
                    st.write(f"""
                    <div style="
                        font-size: 18px;
                        font-weight: bold;
                        margin-top: -5px;
                        margin-bottom: 5px;
                        padding-left: 10px;
                    ">{page['group'].upper()} Scripts</div>""", unsafe_allow_html=True)
                st.page_link(page['page'] , label = page['label'] , icon = page['icon'] , help = page['help'])
                group = page['group']
                # st.switch_page(page['page'])

def main():
    page_config()
    pg = page_navigation()
    sidebar_navigation()
    pg.run()
    
if __name__ == '__main__':
    main() 