"""Streamlit navigation setup for the interactive app.

Provides :func:`page_setup` as the main entry point called from
``launch.py``.  It chains :func:`page_config` (global page settings + CSS),
:func:`top_navigation` (Streamlit MPA header/hidden nav), and
:func:`custom_sidebar_navigation` (logo + per-script sidebar links) depending
on the configured ``navigation_position``.
"""
from __future__ import annotations
import streamlit as st
import streamlit_antd_components as sac
from typing import Literal

from src.proj import Const
from src.interactive.main.util.session_control import SC 

def page_config() -> None:
    """Configure the Streamlit page and apply custom CSS styling.

    Sets the page title, icon, layout, and sidebar state, then calls
    :func:`style` to inject the project's custom CSS.
    """
    st.set_page_config(
        page_title = Const.Pref.interactive.get('page_title' , 'Learndl'),
        page_icon=":material/rocket_launch:",
        layout= 'wide' , # 'centered',
        initial_sidebar_state="expanded"
    )
    from src.interactive.frontend.style import style
    style()

def top_navigation(position : Literal['top',  'hidden'] = 'top'):
    """Build a Streamlit multi-page navigation object grouped by page category.

    Groups intro pages under ``'Intro'`` and script pages under
    ``'<Group> Scripts'``, then passes the result to ``st.navigation``.

    Args:
        position: Where to display the navigation — ``'top'`` or ``'hidden'``.

    Returns:
        The ``st.navigation`` runner returned by Streamlit.
    """
    pages = {}
    intro_only = Const.Pref.interactive.get('navigation_top_intro_only' , True)
    pages[''] = [page['page'] for page in SC.intro_pages.values()]
    for page in SC.script_pages.values():
        group_name = 'hidden' if intro_only else (page['group'].title() + ' Scripts')
        if group_name not in pages: 
            pages[group_name] = []
        pages[group_name].append(page['page'])
    pg = st.navigation(pages = pages , position=position)
    if intro_only:
        # Combine page targets into a single hidden rule block
        hidden_idx = len(pages[''])
        hide_css = f"""
        div[class*="stAppToolbar"] {{
            div[class*="rc-overflow-item"][style*="; order: {hidden_idx};"] {{
                display: none !important; 
            }}
        }}
        [data-testid="stTopNavPopover"] > div > div[data-testid="stMarkdownContainer"]{{
            display: none !important; 
        }}
        [data-testid="stTopNavLink"][href*="/_script_"] {{display: none !important; }}
        [data-testid="stTopNavDropdownLink"][href*="/_script_"] {{display: none !important; }}

        [data-testid="stSidebarNavLink"][href*="/_script_"] {{display: none !important; }}
        ul[data-testid="stSidebarNavItems"] > div {{display: none !important; }}
        """

        # Inject the generated style sheet cleanly
        st.html(f"<style>{hide_css}</style>")

    return pg

def custom_sidebar_navigation() -> None:
    """Render the sidebar logo, app title, and script quick-link panel."""
    with st.sidebar:
        from src.interactive.frontend.logo import get_logo
        st.logo(**get_logo() , link = 'https://github.com/mengkjin/learndl')
        # from src.interactive.main.util.page import PAGE_TITLE
        # st.subheader(f'*_{PAGE_TITLE}_*')
        with st.container(key = "sidebar-quick-links"):
            # intro_links()
            script_links()

def intro_links() -> None:
    """Render icon-button shortcuts for each intro page in the sidebar."""
    pages = SC.intro_pages
    with st.container(key = "sidebar-intro-links"):
        cols = st.columns(len(pages))
        for col , (name , page) in zip(cols , pages.items()):
            if col.button('' , icon = page['icon'] , key = f"sidebar-intro-link-{name}" ,
                          help = f""":blue[**{page['label'].title()}**] - {page['help']}"""):
                st.switch_page(page['page'])

def script_links() -> None:
    """Render per-script run buttons and page links in the sidebar.

    Each script gets a run-readiness indicator button (green/yellow/gray/red)
    and a :func:`st.page_link` label showing its formatted path.

    Args:
        show_dir: If ``True``, insert a group sub-header whenever the script
            group changes.
    """
    from src.interactive.backend.script import ScriptRunner
    with st.container(key = "sidebar-script-links"):
        group = ''
        for name , page in SC.script_pages.items():
            if page['group'] != group:
                with st.container(horizontal= True , vertical_alignment = 'center' , gap = 'xxsmall' , key = f"sidebar-script-links-divider-{page['group']}"):
                    sac.divider(label=page['group'].title() , icon = ':material/folder:', color='gray' , align='center')
            parts : list[str] = page['label'].split(' > ')
            with st.container(horizontal= True , vertical_alignment = 'center' , gap = 'xxsmall'):
                runner : ScriptRunner = page['runner']
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
                with st.container(width = 'stretch'):
                    st.page_link(page['page'] , label = '> '.join([*parts[1:]]) , icon = page['icon'] , help = f":blue[**{parts[0].upper()}**] - {page['help']}" , width = 'stretch')
            group = page['group']

def page_setup(navigation_position : Literal['top', 'sidebar' , 'both'] | None = None):
    """Apply page config and build navigation according to ``navigation_position``.

    - ``'top'``    — top bar only (hidden sidebar nav)
    - ``'sidebar'`` — hidden top nav + custom sidebar nav
    - ``'both'``   — top bar + custom sidebar nav

    Args:
        navigation_position: Controls which navigation elements are rendered.

    Returns:
        The ``st.navigation`` runner; call ``.run()`` to execute the active page.
    """
    if navigation_position is None:
        navigation_position = Const.Pref.interactive.get('navigation_position' , 'both')
    page_config()
    if navigation_position == 'top' or navigation_position == 'both':
        pg = top_navigation('top')
    else:
        pg = top_navigation('hidden')
    if navigation_position == 'sidebar' or navigation_position == 'both':
        custom_sidebar_navigation()
    return pg