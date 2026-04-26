"""Streamlit API console: browse exposed ``src.api`` endpoints and run them."""
from src.interactive.main.util import SC
from src.interactive.main.util.api_adapter import show_api_endpoint_console

PAGE_NAME = "api_console"

@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the streamlit API console page."""
    show_api_endpoint_console()

if __name__ == "__main__":
    main()
