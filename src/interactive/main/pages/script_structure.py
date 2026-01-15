import streamlit as st
import re

from util import SC , get_script_page , print_page_header
from src.interactive.backend import ScriptRunner

PAGE_NAME = 'script_structure'

def show_script_structure():
    """show folder content recursively"""  
    with st.container(key="script-structure-special-expander"):
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
    if runner.script_key not in SC.script_runners: 
        SC.script_runners[runner.script_key] = runner
    
    page = get_script_page(runner.script_key)
    if page is None: 
        return
    
    with st.container(key = f"script-structure-level-{runner.level}-{runner.script_key}"):
        cols = st.columns([1, 1] , gap = "small" , vertical_alignment = "center")
        
        with cols[0]:
            button_text = ':no_entry:' if runner.header.disabled else ':snake:' + ' ' + runner.desc
            widget_key = f"script-runner-expand-{runner.script_key}"
            if st.button(f"**{button_text}**" , key=widget_key , 
                        help = f"*{str(runner.script)}*"):
                st.switch_page(page['page'])
        with cols[1]:
            st.info(f"**{runner.content}**" , icon = ":material/info:")

def main():
    print_page_header(PAGE_NAME)
    show_script_structure() 

if __name__ == '__main__':
    main() 