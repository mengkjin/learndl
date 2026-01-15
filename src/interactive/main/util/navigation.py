import streamlit as st

from .control import SC 
from .page import intro_pages , script_pages
from .logo import get_logo

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
        
        with st.container(key = "sidebar-quick-links"):
            intro_links()
            script_links()

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
                    st.button(":material/mode_off_on:", key=f"direct-script-run-button-enabled-{name}" , 
                            help = f"Script **{runner.script_name}** is ready to run directly" , disabled = False ,
                            on_click = SC.click_script_runner_run , args = (runner, None) , type = 'tertiary')
                else:
                    st.button(":material/do_not_disturb:", key=f"direct-script-run-button-disabled-{name}" , 
                              help = f"Script **{runner.script_name}** needs to be configured first" , disabled = True , type = 'tertiary')
            with cols[1]:
                st.page_link(page['page'] , label = ' > '.join([f'**{parts[0].upper()}**' , *parts[1:]]) , icon = page['icon'] , help = page['help'])
            group = page['group']