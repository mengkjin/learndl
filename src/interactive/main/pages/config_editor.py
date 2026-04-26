"""Config editor page: YAML file editor for model and algo config files."""
from src.interactive.main.util import SC
from src.interactive.main.util.components import show_config_editor
    
PAGE_NAME = 'config_editor'

@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the config editor page."""
    show_config_editor()
    
if __name__ == '__main__':
    main() 