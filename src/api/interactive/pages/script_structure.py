"""script structure page"""
from __future__ import annotations
from src.api.interactive.util.session_control import SC
from src.api.interactive.util.components import show_script_structure

__all__ = ['main']

PAGE_NAME = 'script_structure'

@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the script structure page."""
    show_script_structure() 

if __name__ == '__main__':
    main() 