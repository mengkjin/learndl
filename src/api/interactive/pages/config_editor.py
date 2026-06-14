"""Config editor page: YAML file editor for model and algo config files."""
from __future__ import annotations
from src.api.interactive.util.session_control import SC
from src.api.interactive.util.components import show_config_editor
    
__all__ = ['main']

PAGE_NAME = 'config_editor'

@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the config editor page."""
    show_config_editor()
    
if __name__ == '__main__':
    main() 