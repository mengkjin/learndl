
import streamlit as st

from src.proj import PATH
from src.app.frontend.frontend import YAMLFileEditor

from util import SC , set_current_page , show_sidebar_buttons , print_page_header
    
PAGE_NAME = 'config_editor'

def show_config_editor():
    """show config yaml editor"""
    with st.container(key="special-expander-editor"):
        files = [f for sub in ["train" , "trade" , "nn" , "boost"] for f in PATH.conf.joinpath(sub).glob("*.yaml")]
        default_file = PATH.conf.joinpath("train/model.yaml")
        config_editor = YAMLFileEditor('config-editor', file_root=PATH.conf)
        SC.config_editor_state = config_editor.state
        config_editor.show_yaml_editor(files, default_file=default_file)

def main():
    set_current_page(PAGE_NAME)
    print_page_header(PAGE_NAME)  
    show_config_editor()
    show_sidebar_buttons()
    
if __name__ == '__main__':
    main() 