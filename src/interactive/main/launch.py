

import pathlib , os
import streamlit as st
import streamlit_autorefresh as st_autorefresh

from src.proj import Proj
from util import (SC , style , intro_pages , script_pages , runs_page_url , get_logo)

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
    cols = st.columns([4,1] , gap = 'small' , vertical_alignment = 'center')
    with cols[0]: 
        st.session_state['box-title'] = st.container(key = 'box-title')
    with cols[1]: 
        st.session_state['box-main-button'] = st.container(key = 'box-main-button')
    #st.session_state['box-title'].title(f":rainbow[:material/rocket_launch: {__page_title__} (_v{__version__}_)]")
    #if cols[1].button(':rainbow[:material/home:]' , key = 'go-home-button' , help = 'Go to Home Page'): 
    #    st.switch_page(get_intro_page('home')['page'])

def page_navigation():
    pages = {}
    pages['Intro'] = [page['page'] for page in intro_pages().values()]
    for page in script_pages().values():
        group_name = page['group'].title() + ' Scripts'
        if group_name not in pages: 
            pages[group_name] = []
        pages[group_name].append(page['page'])
    pg = st.navigation(pages = pages , position='top')
    return pg

def sidebar_navigation():
    with st.sidebar:
        st.logo(**get_logo() , link = 'https://github.com/mengkjin/learndl')
        global_button()
        global_settings()

        st.subheader(":blue[:material/link: Quick Links]")
        with st.container(key = "sidebar-quick-links"):
            intro_links()
            script_links()

def global_button():
    _ , col0 , col1 , col2 , _ = st.columns([0.2,1,1,1,0.2] , gap = 'small' , vertical_alignment = 'center')
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
        key = 'sidebar-script-latest-task-button'
        if key not in st.session_state:
            st.session_state[key] = st.empty()
        with st.session_state[key]:
            item = SC.get_latest_task_item()
            if item is None:
                st.button(":material/slideshow:", key=f"{key}-disabled-init" , 
                        help = "Please Run a Task First" , disabled = True)
            else:
                if st.button(":material/slideshow:", key=f"{key}-enable-{item.id}" , 
                            help = f":blue[**Show Latest Task**]: {item.id}" , 
                            on_click = SC.click_show_complete_report , args = (item,) ,
                            disabled = False):
                    st.switch_page(runs_page_url(item.script_key))

    with col2:
        key = 'sidebar-refresh-task-queue-button'
        if key not in st.session_state:
            st.session_state[key] = st.empty()
        with st.session_state[key]:
            st.button(":material/refresh:", key=f"{key}-big" , help = "Refresh Task Queue" , 
                      on_click = SC.click_queue_refresh , disabled = False)
            
def global_settings():
    with st.container(key = "sidebar-global-settings").expander('Global Settings' , icon = ':material/settings:'):
        # max verbosity, yes for 10 , no for 0 , None for default (2 if not set), passed to script params
        cols = st.columns([1,1] , gap = 'small' , vertical_alignment = 'center')
        cols[0].markdown(":blue-badge[Max Verbosity]" , 
                         help="""Should use max verbosity or min? Not selected will use default (2 if not set).""")
        cols[1].segmented_control('MaxVB' , ['yes' , 'no'] , default = None , 
                                  key = 'global-settings-maxvb'  , label_visibility = 'collapsed')

        # email notification, yes , no , None for default, passed to script params
        cols = st.columns([1,1] , gap = 'small' , vertical_alignment = 'center')
        cols[0].markdown(":blue-badge[Email Notification]" , 
                         help="""If email after the script is complete? Not selected will use script header value.""")
        cols[1].segmented_control('Email' , ['yes' , 'no'] , default = None , 
                                  key = 'global-settings-email'  , label_visibility = 'collapsed')

        # run mode, shell , os , or default , used in SessionControl.click_script_runner_run()
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

def script_links(show_dir = False):
    with st.container(key = "sidebar-script-links"):
        def subsubheader(x): 
            st.write(f"""
                <div style="
                    font-size: 16px;
                    font-weight: bold;
                    margin-top: 0px;
                    margin-bottom: 5px;
                    padding-left: 10px;
                ">{x}</div>""", unsafe_allow_html=True)
        
        group = ''
        for name , page in script_pages().items():
            if show_dir and page['group'] != group:
                subsubheader(page['group'].upper() + ' Scripts')
            parts : list[str] = page['label'].split(' > ')
            cols = st.columns([1,19] , gap = 'small' , vertical_alignment = 'center')
            runner = page['runner']
            with cols[0].container(key = f"direct-script-run-{name}"):
                if runner.ready:
                    st.button(":material/play_circle:", key=f"direct-script-run-button-enabled-{name}" , 
                            help = f"Script **{runner.script_name}** is ready to run directly" , disabled = False ,
                            on_click = SC.click_script_runner_run , args = (runner, None) , type = 'tertiary')
                else:
                    st.button(":material/do_not_disturb:", key=f"direct-script-run-button-disabled-{name}" , 
                              help = f"Script **{runner.script_name}** needs to be configured first" , disabled = True , type = 'tertiary')
            with cols[1]:
                st.page_link(page['page'] , label = ' > '.join([f'**{parts[0].upper()}**' , *parts[1:]]) , icon = page['icon'] , help = page['help'])
            group = page['group']
  
def main():
    Proj.print_info(script_level = False)
    page_config()
    pg = page_navigation()
    sidebar_navigation()
    pg.run()
    
if __name__ == '__main__':
    main() 