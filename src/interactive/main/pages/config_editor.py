"""Config editor page: YAML file editor for model and algo config files."""
import streamlit as st

from src.proj import PATH
from src.interactive.frontend.frontend import YAMLFileEditor

from src.interactive.main.util import SC , print_page_header
    
PAGE_NAME = 'config_editor'

def show_config_editor() -> None:
    """Render the YAML file editor scoped to model and algo config files.

    Discovers all ``*.yaml`` files under ``configs/model/`` and
    ``configs/algo/``, pre-selects ``configs/model/model.yaml``, and passes
    them to a :class:`YAMLFileEditor` widget.
    """
    with st.container(key="special-expander-editor"):
        files = [f for sub in ["model" , "algo"] for f in PATH.conf.joinpath(sub).rglob("*.yaml")]
        default_file = PATH.conf.joinpath("model" , "model.yaml")
        config_editor = YAMLFileEditor('config-editor', file_root=PATH.conf)
        SC.config_editor_state = config_editor.state
        config_editor.show_yaml_editor(files, default_file=default_file)

def main() -> None:
    """Entry point for the config editor page."""
    print_page_header(PAGE_NAME)
    show_config_editor()
    
if __name__ == '__main__':
    main() 