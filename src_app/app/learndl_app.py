

import sys , pathlib , os
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl' , 1)[-1])
if not path in sys.path: sys.path.append(path)
assert os.getcwd() == path , \
    f'current working directory {os.getcwd()} is not {path} , do not know where to find src file'

import streamlit as st
import streamlit_autorefresh as st_autorefresh

from util import (__page_title__ , __version__ , SC , AUTO_REFRESH_INTERVAL , 
                  style , intro_pages , script_pages , runs_page_url)

st.set_option('client.showSidebarNavigation', False)

if AUTO_REFRESH_INTERVAL > 0:
    st_autorefresh.st_autorefresh(interval = AUTO_REFRESH_INTERVAL * 1000)

def page_config():
    st.set_page_config(
        page_title=__page_title__,
        page_icon=":material/rocket_launch:",
        layout= 'wide' , # 'centered',
        initial_sidebar_state="expanded"
    )
    style()
    cols = st.columns([4,1] , gap = 'small' , vertical_alignment = 'center')
    with cols[0]: st.session_state['box_title'] = st.empty()
    with cols[1]: st.session_state['box_main_button'] = st.empty()
    #st.session_state['box_title'].title(f":rainbow[:material/rocket_launch: {__page_title__} (_v{__version__}_)]")
    #if cols[1].button(':rainbow[:material/home:]' , key = 'go-home-button' , help = 'Go to Home Page'): 
    #    st.switch_page(get_intro_page('home')['page'])

def page_navigation():
    pages = {}
    pages['Intro'] = [page['page'] for page in intro_pages().values()]
    for page in script_pages().values():
        group_name = page['group'].title() + ' Scripts'
        if group_name not in pages: pages[group_name] = []
        pages[group_name].append(page['page'])
    pg = st.navigation(pages = pages , position='top')
    return pg


def sidebar_navigation():
    with st.sidebar:
        st.logo(pathlib.Path(file_path).parent / "images/image.png" ,
                icon_image=pathlib.Path(file_path).parent / "images/icon_image.png")
        
        global_button()
        global_settings()

        st.subheader(":blue[:material/link: Quick Links]")
        with st.container(key = "sidebar-quick-links"):
            intro_links()
            script_links()

def global_button():
    _ , col0 , col1 , _ = st.columns([0.5,1,1,0.5] , gap = 'small' , vertical_alignment = 'center')
    with col0:
        key = 'sidebar-runner-button'
        if key not in st.session_state:
            st.session_state[key] = st.empty()
        with st.session_state[key]:
            help_text = f"Please Choose a Script to Run First"
            button_key = f"script-runner-run-disabled-init-sidebar"
            st.button(":material/mode_off_on:", key=button_key , 
                    help = help_text , disabled = True)
            
    with col1:
        key = 'sidebar-latest-task-button'
        if key not in st.session_state:
            st.session_state[key] = st.empty()
        with st.session_state[key]:
            item = SC.get_latest_task_item()
            if item is None:
                st.button(":material/slideshow:", key=f"script-latest-task-disabled-init" , 
                        help = "Please Run a Task First" , disabled = True)
            else:
                if st.button(":material/slideshow:", key=f"script-latest-task-enable-{item.id}" , 
                            help = f"Show Task {item.id}" , 
                            on_click = SC.click_show_complete_report , args = (item,) ,
                            disabled = False):
                    st.switch_page(runs_page_url(str(item.relative)))
            
def global_settings():
    with st.container(key = "sidebar-global-settings").expander('Global Settings' , icon = ':material/settings:'):
        cols = st.columns([1,1] , gap = 'small' , vertical_alignment = 'center')
        cols[0].markdown(":blue-badge[Email Notification]" , 
                         help="""If email after the script is complete? Not selected will use script header value.""")
        cols[1].segmented_control('Email' , ['Y' , 'N'] , default = None , 
                                  key = 'global-settings-email'  , label_visibility = 'collapsed')
        cols = st.columns([1,1] , gap = 'small' , vertical_alignment = 'center')
        cols[0].markdown(":blue-badge[Run Mode]" , 
                         help='''Which mode should the script be running in?
                           :blue[**shell**] will start a commend terminal to run;
                           :blue[**os**] will run in backend.
                           Not selected will use script header value.''')
        cols[1].segmented_control('Mode' , ['shell' , 'os'] , default = None , 
                                  key = 'global-settings-mode' , label_visibility = 'collapsed')

def intro_links():
    pages = intro_pages()
    with st.container(key = "sidebar-intro-links"):
        cols = st.columns(len(pages))
        for col , (name , page) in zip(cols , pages.items()):
            if col.button('' , icon = page['icon'] , key = f"sidebar-intro-link-{name}" ,
                          help = f""":blue[**{page['label'].title()}**] - {page['help']}"""):
                st.switch_page(page['page'])

def script_links():
    with st.container(key = "sidebar-script-links"):
        subsubheader = lambda x: st.write(f"""
                <div style="
                    font-size: 16px;
                    font-weight: bold;
                    margin-top: 0px;
                    margin-bottom: 5px;
                    padding-left: 10px;
                ">{x}</div>""", unsafe_allow_html=True)
        
        group = ''
        for page in script_pages().values():
            if page['group'] != group:
                subsubheader(page['group'].upper() + ' Scripts')
            st.page_link(page['page'] , label = page['label'] , icon = page['icon'] , help = page['help'])
            group = page['group']
  
def main():
    page_config()
    pg = page_navigation()
    sidebar_navigation()
    pg.run()
    
if __name__ == '__main__':
    main() 