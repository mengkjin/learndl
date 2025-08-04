
import streamlit as st

from src_app.db import CONF_DIR

from src_app.frontend.frontend import YAMLFileEditor

from util import starter , SC
    
def show_config_editor():
    """show config yaml editor"""
    files = [f for sub in ["train" , "trade" , "nn" , "boost"] for f in CONF_DIR.joinpath(sub).glob("*.yaml")]
    default_file = CONF_DIR.joinpath("train/model.yaml")

    container = st.container(key="special-expander-editor")
    with container:
        st.header(":material/edit_document: Config File Editor" , divider = 'grey')
        st.info("This File Editor is for editing selected config files" , icon = ":material/info:")
        st.info("For other config files, please use the file explorer" , icon = ":material/info:")
        
        config_editor = YAMLFileEditor('config-editor', file_root=CONF_DIR)
        SC.config_editor_state = config_editor.state
        config_editor.show_yaml_editor(files, default_file=default_file)
    
if __name__ == '__main__':
    starter()
    show_config_editor() 