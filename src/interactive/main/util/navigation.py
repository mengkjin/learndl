"""Streamlit navigation setup for the interactive app.

Provides :func:`page_setup` as the main entry point called from
``launch.py``.  It chains :func:`page_config` (global page settings + CSS),
:func:`top_navigation` (Streamlit MPA header/hidden nav), and
:func:`custom_sidebar_navigation` (logo + per-script sidebar links) depending
on the configured ``navigation_position``.
"""
from __future__ import annotations
import streamlit as st
from typing import Any, Literal

from src.proj import CONST
from src.interactive.frontend.logo import get_logo
from src.interactive.frontend.style import style
from src.interactive.backend.script import ScriptRunner
from .control import SC 
from .page import intro_pages , script_pages , PAGE_TITLE

def page_config() -> None:
    """Configure the Streamlit page and apply custom CSS styling.

    Sets the page title, icon, layout, and sidebar state, then calls
    :func:`style` to inject the project's custom CSS.
    """
    st.set_page_config(
        page_title=CONST.Pref.get('interactive' , 'page_title' , 'Learndl'),
        page_icon=":material/rocket_launch:",
        layout= 'wide' , # 'centered',
        initial_sidebar_state="expanded"
    )
    style()

def top_navigation(position : Literal['top', 'sidebar' , 'hidden'] = 'top') -> Any:
    """Build a Streamlit multi-page navigation object grouped by page category.

    Groups intro pages under ``'Intro'`` and script pages under
    ``'<Group> Scripts'``, then passes the result to ``st.navigation``.

    Args:
        position: Where to display the navigation — ``'top'``, ``'sidebar'``,
            or ``'hidden'``.

    Returns:
        The ``st.navigation`` runner returned by Streamlit.
    """
    pages = {}
    pages['Intro'] = [page['page'] for page in intro_pages().values()]
    for page in script_pages().values():
        group_name = page['group'].title() + ' Scripts'
        if group_name not in pages: 
            pages[group_name] = []
        pages[group_name].append(page['page'])
    pg = st.navigation(pages = pages , position=position)
    return pg

def custom_sidebar_navigation() -> None:
    """Render the sidebar logo, app title, and script quick-link panel."""
    with st.sidebar:
        st.logo(**get_logo() , link = 'https://github.com/mengkjin/learndl')
        st.subheader(f'*_{PAGE_TITLE}_*')
        with st.container(key = "sidebar-quick-links"):
            # intro_links()
            script_links()

def intro_links() -> None:
    """Render icon-button shortcuts for each intro page in the sidebar."""
    pages = intro_pages()
    with st.container(key = "sidebar-intro-links"):
        cols = st.columns(len(pages))
        for col , (name , page) in zip(cols , pages.items()):
            if col.button('' , icon = page['icon'] , key = f"sidebar-intro-link-{name}" ,
                          help = f""":blue[**{page['label'].title()}**] - {page['help']}"""):
                st.switch_page(page['page'])

def script_links(show_dir: bool = False) -> None:
    """Render per-script run buttons and page links in the sidebar.

    Each script gets a run-readiness indicator button (green/yellow/gray/red)
    and a :func:`st.page_link` label showing its formatted path.

    Args:
        show_dir: If ``True``, insert a group sub-header whenever the script
            group changes.
    """
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
            cols = st.columns([1,19] , gap = 'xxsmall' , vertical_alignment = 'center')
            runner : ScriptRunner = page['runner']
            with cols[0].container(key = f"direct-script-run-{name}"):
                if runner.ready == 3:
                    st.button(":material/mode_off_on:", key=f"direct-script-run-button-enable-green-{name}" , 
                            help = f"Script **{runner.script_name}** is ready to run directly" , disabled = False ,
                            on_click = SC.click_script_runner_run , args = (runner, None) , type = 'tertiary')
                elif runner.ready == 2:
                    st.button(":material/mode_off_on:", key=f"direct-script-run-button-enable-yellow-{name}" , 
                            help = f"Script **{runner.script_name}** is ready to run directly (parameters use defaults)" , disabled = False ,
                            on_click = SC.click_script_runner_run , args = (runner, None) , type = 'tertiary')
                elif runner.ready == 1:
                    st.button(":material/enable:", key=f"direct-script-run-button-disable-gray-{name}" , 
                              help = f"Script **{runner.script_name}** needs to be configured first" , disabled = True , type = 'tertiary')
                else:
                    st.button(":material/do_not_disturb:", key=f"direct-script-run-button-disable-red-{name}" , 
                              help = f"Script **{runner.script_name}** is disabled or blacklisted on this machine" , disabled = True , type = 'tertiary')
            with cols[1]:
                st.page_link(page['page'] , label = ' > '.join([f'**{parts[0].upper()}**' , *parts[1:]]) , icon = page['icon'] , help = page['help'])
            group = page['group']

def page_setup(navigation_position : Literal['top', 'sidebar' , 'both'] = CONST.Pref.get('interactive' , 'navigation_position' , 'both')) -> Any:
    """Apply page config and build navigation according to ``navigation_position``.

    - ``'top'``    — top bar only (hidden sidebar nav)
    - ``'sidebar'`` — hidden top nav + custom sidebar nav
    - ``'both'``   — top bar + custom sidebar nav

    Args:
        navigation_position: Controls which navigation elements are rendered.

    Returns:
        The ``st.navigation`` runner; call ``.run()`` to execute the active page.
    """
    page_config()
    if navigation_position == 'top' or navigation_position == 'both':
        pg = top_navigation('top')
    else:
        pg = top_navigation('hidden')
    if navigation_position == 'sidebar' or navigation_position == 'both':
        custom_sidebar_navigation()
    return pg