"""Page registry, page-file generation, and page-header rendering.

Maintains the static lists of intro pages and dynamically discovered script
    get_intro_page , get_script_page , print_page_header ,
    show_script_detail , show_param_settings , show_report_main) 
each page's ``main()``.
"""
import re
import streamlit as st
from pathlib import Path
from typing import Literal

from src.proj import Proj , Const

from .control import SC , set_current_page

PAGE_DIR = Path(__file__).parent.parent.joinpath('pages')
assert PAGE_DIR.exists() , f"Page directory {PAGE_DIR} does not exist"

INTRO_PAGES = ['home' , 'developer_info' , 'config_editor' , 'task_queue']

PAGE_TITLE = f":red[:material/rocket_launch:] :rainbow[{Const.Pref.interactive.get('page_title' , 'Learndl')} (_v{Proj.version}_)]"

PAGE_ICONS = {
    'home' : ':material/home:' ,
    'developer_info' : ':material/bug_report:' ,
    'config_editor' : ':material/edit_document:' ,
    'task_queue' : ':material/event_list:' ,
    'script_structure' : ':material/account_tree:' ,
}

PAGE_HELPS = {
    'home' : f"Tutorial , System Info and Links." ,
    'developer_info' : f"This is for developer only. Check boxes to select what information to show." ,
    'config_editor' : 'This File Editor is for editing selected config files. For other config files, please use the file explorer.' ,
    'task_queue' : f"Shows the entire task queue. Adjust filter to show more specific tasks." ,
    'script_structure' : f"The script structure of project runs. Click the script button to switch to script page." ,
}

SCRIPT_ICONS = {
    'check'   : ':material/question_mark:' ,
    'autorun' : ':material/schedule:' ,
    'data'    : ':material/database:' ,
    'factor'  : ':material/graph_3:' ,
    'train'   : ':material/model_training:' ,
    'test'    : ':material/experiment:' ,
    'predict' : ':material/online_prediction:' ,
    'trading' : ':material/payments:',
}


def script_detail_url_path(script_key: str) -> str:
    """Stable ``url_path`` for each script's ``st.Page`` (must be unique per script).

    Streamlit forbids ``/`` in ``url_path`` (no nested segments); use a single token.
    """
    slug = re.sub(r'[/\\]', '_', script_key)
    return f'_script_{slug}'


def script_detail_page_callable(script_key: str):
    """Return a no-arg callable suitable for ``st.Page(..., url_path=...)``."""
    def _run() -> None:
        from src.interactive.main.util.script_detail import show_script_detail
        show_script_detail(script_key)
    return _run

def intro_pages() -> dict[str, dict]:
    """Return a dict mapping intro page name → page metadata for all intro pages."""
    return {page:get_intro_page(page) for page in INTRO_PAGES}

def get_intro_page(page_name : str) -> dict:
    """Return (and cache) the metadata dict for a single intro page.

    The result is stored in ``st.session_state['app_intro_pages']`` so that
    the ``st.Page`` object is only created once per session.

    Args:
        page_name: Must be one of :data:`INTRO_PAGES`.

    Returns:
        Dict with keys ``page``, ``label``, ``head``, ``icon``, ``help``.
    """
    assert page_name in INTRO_PAGES , f"Page {page_name} not a valid intro page"
    if 'app_intro_pages' not in st.session_state: 
        st.session_state['app_intro_pages'] = {}
    if page_name not in st.session_state['app_intro_pages']:
        label = page_name.replace('_', ' ').title()
        icon = PAGE_ICONS[page_name]
        help = PAGE_HELPS[page_name]
        st.session_state['app_intro_pages'][page_name] = {
            'page' : st.Page(f'pages/{page_name}.py' , title = label , icon = icon) ,
            'label' : label ,
            'head' : label ,
            'icon' : icon ,
            'help' : help ,
        }
    return st.session_state['app_intro_pages'][page_name]

def script_pages() -> dict[str, dict]:
    """Return a dict mapping script key → page metadata for all enabled scripts.

    Iterates discovered :class:`PathItem` file entries and caches each
    ``st.Page`` in session state (callable-backed; no stub ``.py`` files).
    """
    pages = {}
    items = [item for item in SC.path_items if item.is_file and item.level > 0]
    for item in items:
        pages[item.script_key] = get_script_page(item.script_key)
    return pages

def get_script_page(script_key: str) -> dict:
    """Return (and cache) the metadata dict for a script page.

    Creates a ``st.Page`` wrapping :func:`~src.interactive.main.util.script_detail.show_script_detail`
    and stores the result in ``st.session_state['app_script_pages']``.

    Args:
        script_key: Relative path key such as ``'4_train/1_train_model.py'``.

    Returns:
        Dict with keys ``page``, ``group``, ``label``, ``head``, ``icon``,
        ``help``, ``runner``.
    """
    runner = SC.get_script_runner(script_key)
    if runner.header.disabled: 
        st.error(f"Script {script_key} is disabled!")
        return {}
    if 'app_script_pages' not in st.session_state: 
        st.session_state['app_script_pages'] = {}
    
    if runner.script_key not in st.session_state['app_script_pages']:
        if runner.script_key not in SC.script_runners: 
            SC.script_runners[runner.script_key] = runner
        
        icon = SCRIPT_ICONS[runner.script_group]
        help = f"**Script**: *{str(runner.script)}*\n**Description**: {runner.content}"
        if runner.todo: 
            help += f"\n**TODO**: {runner.todo}"
        st.session_state['app_script_pages'][runner.script_key] = {
            'page' : st.Page(
                script_detail_page_callable(runner.script_key),
                title = runner.format_path,
                icon = icon,
                url_path = script_detail_url_path(runner.script_key),
            ),
            'group' : runner.script_group ,
            'label' : runner.format_path ,
            'head' : runner.format_path ,
            'icon' : icon ,
            'help' : help ,
            'runner' : runner ,
        }
    return st.session_state['app_script_pages'][runner.script_key]

def print_page_header(page_name : str , type : Literal['intro' , 'script'] = 'intro') -> None:
    """Register the active page and render the shared header and control panel.

    Called at the top of every page's ``main()`` function.  Sets the current
    page in session state, notifies :class:`SessionControl`, then renders the
    coloured icon + rainbow title header followed by the control panel row.

    Args:
        page_name: The intro page name (e.g. ``'home'``) or script key
            (e.g. ``'4_train/1_train_model.py'``).
        type: ``'intro'`` for intro pages, ``'script'`` for script pages.
    """
    set_current_page(page_name)
    SC.switch_page(page_name)
    if type == 'intro':
        script_key = None
        self_page = get_intro_page(page_name) 
    elif type == 'script':
        script_key = page_name
        self_page = get_script_page(script_key)
        if self_page is None:
            st.error(f"Script {script_key} not not enabled")
            return
    else:
        raise ValueError(f"type {type} should be 'intro' or 'script'")
    
    # st.title(PAGE_TITLE)
    st.header(f"*_:red[{self_page['icon']}] :rainbow[{self_page['head']}]_*" , help = self_page['help'])
    # if 'control-panel' not in st.session_state:
    #     st.session_state['control-panel'] = ControlPanel()
    # st.session_state['control-panel'].show(script_key = script_key)
    SC.get_control_panel().show(script_key = script_key)
    