import streamlit as st
import re

from menu import starter
from control import SC    
from src_ui.backend import ScriptRunner

def show_script_structure():
    """show folder content recursively"""  
    st.header(":blue[:material/folder_open: Script Structure]" , divider = 'grey')
    items = SC.path_items
    for item in items:
        if item.is_dir:
            folder_name = re.sub(r'^\d+_', '', item.name).replace('_', ' ').title()
            body = f"""
            <div style="
                font-size: 18px;
                font-weight: bold;
                margin-top: 5px;
                margin-bottom: 5px;
                letter-spacing: 3px;
                margin-left: {(item.level)*45}px;
            ">📂 {folder_name}</div>
            """       
            st.markdown(body , unsafe_allow_html=True)
 
        elif item.level > 0:
            show_script_runner(item.script_runner())

def show_script_runner(runner: ScriptRunner):
    """show single script runner"""
    SC.script_runners[runner.script_key] = runner
    with st.container(key = f"script-structure-level-{runner.level}-{runner.script_key}"):
        cols = st.columns([1, 1] , gap = "small" , vertical_alignment = "center")
        
        with cols[0]:
            button_text = ':no_entry:' if runner.header.disabled else ':snake:' + ' ' + runner.desc
            selected = SC.current_script_runner is not None and SC.current_script_runner == runner.script_key
            widget_key = f"script-runner-expand-{runner.script_key}" if not selected else f"script-runner-expand-selected-{runner.script_key}"
            if st.button(f"**{button_text}**" , key=widget_key , 
                        help = f"*{str(runner.script)}*" ,
                        on_click = SC.click_script_runner_expand , args = (runner,)):
                st.switch_page(f'pages/{runner.script_key.replace("/", "_")}')
        with cols[1]:
            st.info(f"**{runner.content}**" , icon = ":material/info:")

if __name__ == '__main__':
    starter()
    show_script_structure() 