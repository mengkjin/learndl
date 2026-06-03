"""Home page: tutorial, system info, and pending-features banner."""

from src.interactive.main.util.session_control import SC
from src.interactive.main.util.components import show_pending_features # show_tutorial  , show_system_info
from src.interactive.main.util.quick_calls.main import show_quick_calls

PAGE_NAME = 'home'
   
@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the home page."""
    # show_tutorial()
    # show_system_info()
    show_pending_features()
    show_quick_calls()
    
if __name__ == '__main__':
    main() 