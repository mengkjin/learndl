"""Home page: tutorial, system info, and pending-features banner."""
from src.interactive.main.util import SC
from src.interactive.main.util.components import show_system_info , show_pending_features , show_tutorial

PAGE_NAME = 'home'
 
@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the home page."""
    show_tutorial()
    show_system_info()
    show_pending_features()
    
if __name__ == '__main__':
    main() 