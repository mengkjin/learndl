
import streamlit as st

from src.proj import PATH
from src.interactive.frontend.frontend import YAMLFileEditor

from util import SC , print_page_header
    
PAGE_NAME = 'config_editor'

def show_config_editor():
    """show config yaml editor"""
    with st.container(key="special-expander-editor"):
        files = [f for sub in ["model" , "algo" , "schedule"] for f in PATH.conf.joinpath(sub).rglob("*.yaml")]
        default_file = PATH.conf.joinpath("model" , "model.yaml")
        config_editor = YAMLFileEditor('config-editor', file_root=PATH.conf)
        SC.config_editor_state = config_editor.state
        config_editor.show_yaml_editor(files, default_file=default_file)

def main():
    print_page_header(PAGE_NAME)
    show_config_editor()
    
if __name__ == '__main__':
    main() 