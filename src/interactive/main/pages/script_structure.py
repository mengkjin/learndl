"""script structure page"""
from __future__ import annotations
from src.interactive.main.util.session_control import SC
from src.interactive.main.util.components import show_script_structure

PAGE_NAME = 'script_structure'

@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the script structure page."""
    show_script_structure() 

if __name__ == '__main__':
    main() 