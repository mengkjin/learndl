import streamlit as st
import re

from src_app.backend import ScriptRunner
from .control import SC
from .basic import __version__ , __recommended_explorer__ , __page_title__
from .style import style

def menu():
    st.subheader(":material/menu: Menu")
    st.page_link("home.py" , label = "Home" , icon = ":material/home:")
    st.page_link("pages/developer_info.py" , label = "Developer Info" , icon = ":material/bug_report:")
    st.page_link("pages/config_editor.py" , label = "Config Editor" , icon = ":material/edit_document:")
    st.page_link("pages/task_queue.py" , label = "Task Queue" , icon = ":material/event_list:")
    st.page_link("pages/script_structure.py" , label = "Script Structure" , icon = ":material/folder_open:")
    st.subheader(":material/file_present: Scripts")


def script_menu():
    """show folder content recursively"""
    items = SC.path_items

    for item in items:
        if not item.is_dir and item.level > 0:
            runner = item.script_runner()
            show_script_runner(runner)
                    
def show_script_runner(runner: ScriptRunner):
    """show single script runner"""
    if runner.header.disabled: return
    SC.initialize()
    SC.script_runners[runner.script_key] = runner
    with st.container(key = f"script-container-level-{runner.level}-{runner.script_key}"):
        selected = SC.current_script_runner is not None and SC.current_script_runner == runner.script_key
        widget_key = f"script-runner-expand-{runner.script_key}" if not selected else f"script-runner-expand-selected-{runner.script_key}"
        # st.button(runner.desc , key=widget_key ,  icon = ':material/code:' ,
        #           help = f":material/info: **{runner.content}** \n*{str(runner.script)}*" ,
        #           on_click = SC.click_script_runner_expand , args = (runner,) , type = 'tertiary')
        st.page_link(f'pages/{re.sub(r'[/\\]', '_', runner.script_key)}' , 
                     label = runner.format_path ,
                     icon = ':material/code:' ,
                     help = f":material/info: **{runner.content}** \n*{str(runner.script)}*")
        
def starter():
    style()
    with st.sidebar:
        menu()
        script_menu()
    st.title(f":rainbow[:material/rocket_launch: {__page_title__} (_v{__version__}_)]")
    