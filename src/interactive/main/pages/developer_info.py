"""Developer info page: session state, task queue JSON, and log management."""
from src.interactive.main.util import SC
from src.interactive.main.util.components import show_developer_info

PAGE_NAME = 'developer_info'
    
@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the developer info page."""
    show_developer_info()

if __name__ == '__main__':
    main()