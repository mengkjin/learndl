"""Developer info page: session state, task queue JSON, and log management."""
from __future__ import annotations
from src.api.interactive.util.session_control import SC
from src.api.interactive.util.components import show_developer_info

__all__ = ['main']

PAGE_NAME = 'developer_info'
    
@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the developer info page."""
    show_developer_info()

if __name__ == '__main__':
    main()