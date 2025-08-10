import streamlit as st
import re

from util import SC , set_current_page , show_run_button_sidebar
from src_app.backend import ScriptRunner

def show_script_structure():
    """show folder content recursively"""  
    container = st.container(key="script-structure-special-expander")
    with container:
        st.header(":material/folder_open: Script Structure" , divider = 'grey')
        st.info("The script structure of project runs" , icon = ":material/info:")
        st.info("Click the script button to switch to page of the script" , icon = ":material/info:")
        
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
    if runner.script_key not in SC.script_runners: SC.script_runners[runner.script_key] = runner
    with st.container(key = f"script-structure-level-{runner.level}-{runner.script_key}"):
        cols = st.columns([1, 1] , gap = "small" , vertical_alignment = "center")
        
        with cols[0]:
            button_text = ':no_entry:' if runner.header.disabled else ':snake:' + ' ' + runner.desc
            widget_key = f"script-runner-expand-{runner.script_key}"
            if st.button(f"**{button_text}**" , key=widget_key , 
                        help = f"*{str(runner.script)}*"):
                assert SC.script_pages is not None , "SC.script_pages is not initialized"
                st.switch_page(SC.script_pages[runner.script_key]['page'])
        with cols[1]:
            st.info(f"**{runner.content}**" , icon = ":material/info:")

def main():
    set_current_page("script_structure")
    show_script_structure() 
    show_run_button_sidebar()

if __name__ == '__main__':
    main() 